import argparse
import http.client
import json
import math
import random
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ========== 参数设置 ==========
# 模拟器 HTTP 接口地址
server_base_url = "http://127.0.0.1:2026"
# 参赛队号/机器人编号
team_robot_id = ""  #空字符串，运行时通过--robot-id传入

# 竞技场半径，单位：米
field_radius = 1800.0
# 单次测量最远能感知的距离，单位：米
max_measure_range = 1500.0
# 题目给出的清除半径参考值
clear_radius = 20.0
# 真正执行清除时使用的安全半径，略小于 20 米
safe_clear_radius = 19.4
# 示向度测量误差，转成弧度
bearing_angle_error = math.radians(1.006)

# 所有可用频道：1 到 20
all_channels = range(1, 21)
# 题目最多 16 个干扰源
max_source_count = 16
# 一整圈对应的弧度
full_circle = 2.0 * math.pi

# 单次 HTTP 请求默认超时时间
default_timeout_seconds = 4.0
# 单次 HTTP 请求默认重试次数
default_retry_times = 2
# 虚拟时间上限
max_virtual_seconds_limit = 360000.0
# 退出前预留的现实时间，避免来不及调用 /exit
exit_reserve_seconds = 3.0

# 单个频道最多定位尝试次数
max_localize_try_count = 220
# 主循环最多跑多少轮
max_main_loop_rounds = 100000

# 竞技场外允许探测的半径上限
outside_probe_margin = 260.0

# 缺失判定阈值：必须多方向、多半径都没有信号，才认为该频道不存在
absent_min_no_signal_count = 260    # 至少 260 次 no_signal
absent_min_angle_sector_count = 7   # 至少覆盖 12 个角度扇区中的 7 个
absent_min_radius_band_count = 3    # 至少覆盖 3 个径向带


# ========== 基础工具 ==========
# 计算两个二维点之间的直线距离
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# 模型/几何约束出现问题时抛出的异常
class ModelError(RuntimeError):
    pass


# HTTP 响应不符合协议时抛出的异常
class ProtocolError(RuntimeError):
    pass


# 网络连接、超时、响应丢失时抛出的异常
class TransportError(RuntimeError):
    pass


# 时间预算或虚拟时间预算不足时抛出的异常
class BudgetError(RuntimeError):
    pass


# ========== 几何剪裁 ==========
# 用一个半平面 nx*x + ny*y <= c 去裁剪多边形 poly
def clip_halfplane(poly, nx, ny, c):
    if not poly:
        return []
    result = []
    a = poly[-1]
    da = nx * a[0] + ny * a[1] - c
    for b in poly:
        db = nx * b[0] + ny * b[1] - c
        a_in, b_in = da <= 1e-9, db <= 1e-9
        if a_in != b_in:
            t = max(0.0, min(1.0, da / (da - db)))
            result.append((a[0] + t * (b[0] - a[0]),
                           a[1] + t * (b[1] - a[1])))
        if b_in:
            result.append(b)
        a, da = b, db
    clean = []
    for p in result:
        if not clean or distance(p, clean[-1]) > 1e-8:
            clean.append(p)
    if len(clean) > 1 and distance(clean[0], clean[-1]) < 1e-8:
        clean.pop()
    return clean


# 生成一个圆盘的外接近似多边形，用于表示初始可行域
def outer_disk(center, radius, n=72):
    rv = radius / math.cos(math.pi / n)
    return [(center[0] + rv * math.cos(full_circle * i / n),
             center[1] + rv * math.sin(full_circle * i / n))
            for i in range(n)]


# 用“距离中心不超过 radius”的圆盘去裁剪多边形
def clip_disk_outer(poly, center, radius, n=48):
    for i in range(n):
        a = full_circle * (i + 0.5) / n
        nx, ny = math.cos(a), math.sin(a)
        poly = clip_halfplane(poly, nx, ny,
                              radius + nx * center[0] + ny * center[1])
        if not poly:
            return poly
    return poly


# 根据一次示向度测量，把可行域裁剪成两条边界射线之间的角域
def clip_bearing(poly, position, degrees):
    a = math.radians(degrees)
    lo, hi = a - bearing_angle_error, a + bearing_angle_error
    for nx, ny in ((math.sin(lo), -math.cos(lo)),
                   (-math.sin(hi), math.cos(hi))):
        poly = clip_halfplane(poly, nx, ny,
                              nx * position[0] + ny * position[1])
        if not poly:
            return poly
    return poly


# ========== 最小包围圆 ==========
# 以线段 ab 为直径构造圆，返回圆心和半径
def _diameter_circle(a, b):
    c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return c, distance(a, b) / 2


# 求经过三个点的圆，返回圆心和半径
def _three_circle(a, b, c):
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    d = 2 * (bx * cy - by * cx)
    if abs(d) < 1e-10:
        pairs = [_diameter_circle(a, b), _diameter_circle(a, c),
                 _diameter_circle(b, c)]
        return max(pairs, key=lambda z: z[1])
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    p = (a[0] + (cy * b2 - by * c2) / d,
         a[1] + (bx * c2 - cx * b2) / d)
    return p, distance(p, a)


# 求多边形的最小包围圆，随机种子固定，保证结果稳定
def enclosing_circle(poly):
    if not poly:
        raise ModelError("定位可行域为空")
    pts = list(poly)
    random.Random(20260911).shuffle(pts)
    center, radius = pts[0], 0.0
    for i, a in enumerate(pts):
        if distance(a, center) <= radius + 1e-8:
            continue
        center, radius = a, 0.0
        for j, b in enumerate(pts[:i]):
            if distance(b, center) <= radius + 1e-8:
                continue
            center, radius = _diameter_circle(a, b)
            for c in pts[:j]:
                if distance(c, center) > radius + 1e-8:
                    center, radius = _three_circle(a, b, c)
    radius = max(distance(center, p) for p in poly) + 1e-7
    return center, radius


# ========== 单频道状态 ==========
# 记录一个频道的状态：
# unknown：还不知道有没有目标
# tracked：已发现目标，正在定位
# cleared：已经成功清除
# stuck：暂时卡住，定位或清除失败
# absent：多方向 no_signal，认为不存在
class Track:
    def __init__(self, channel):
        self.channel = channel
        self.status = "unknown"
        self.poly = []
        self.observations = []
        self.no_signal_history = []
        self.near_point = None

    # 返回当前可行域的最小包围圆：(圆心, 半径)
    def circle(self):
        if self.near_point is not None:
            return self.near_point, 5.0
        if not self.poly:
            return (0.0, 0.0), field_radius
        return enclosing_circle(self.poly)


# ========== HTTP 传输 ==========
# 负责和模拟器进行 HTTP 通信，所有请求都是 POST JSON，带超时和重试
class HttpTransport:
    def __init__(self, base_url, timeout=default_timeout_seconds,
                 retries=default_retry_times):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    # 向模拟器发送一次请求，返回解析后的 JSON 字典
    def request(self, path, payload, deadline=None):
        body = json.dumps(payload, ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
        last = None
        for attempt in range(self.retries + 1):
            left = (math.inf if deadline is None
                    else deadline - time.monotonic())
            if left <= 0.05:
                raise BudgetError("请求截止时间已到。")
            request = Request(self.base_url + path, data=body,
                              headers={"Content-Type": "application/json"},
                              method="POST")
            try:
                with urlopen(request,
                             timeout=min(self.timeout, left)) as response:
                    status = response.status
                    raw = response.read()
            except HTTPError as e:
                raise ProtocolError("HTTP %s: %s" % (
                    e.code,
                    e.read().decode("utf-8", "replace")[:400])) from e
            except (URLError, OSError, TimeoutError,
                    http.client.HTTPException) as e:
                last = e
                if attempt == self.retries:
                    break
                delay = 0.15 * (2 ** attempt)
                if deadline is not None:
                    delay = max(0, min(delay,
                                       deadline - time.monotonic()))
                time.sleep(delay)
                continue
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError) as e:
                raise ProtocolError("响应不是有效JSON") from e
            if status != 200 or not isinstance(result, dict):
                raise ProtocolError("HTTP状态或响应结构异常。")
            return result
        raise TransportError("连接失败或响应丢失：%s" % last)


# ========== 客户端 ==========
# 对模拟器接口的封装，负责维护当前位置、频道、虚拟时间、现实截止时间、日志和统计指标
class Client:
    def __init__(self, transport, robot_id, log_path=None):
        self.transport = transport
        self.robot_id = robot_id
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_time = 0.0
        self.max_virtual = max_virtual_seconds_limit
        self.started_at = None
        self.deadline = None
        self.exit_reserve = exit_reserve_seconds
        self.closed = False
        self.counter = 0
        self.prefix = uuid.uuid4().hex[:12]
        self.metrics = dict(distance_m=0.0, measurements=0,
                            channel_switches=0, clear_attempts=0,
                            cleared=0, failed_clears=0)
        self.log = (open(log_path, "w", encoding="utf-8")
                    if log_path else None)

    # 检查现实时间和虚拟时间是否还够用
    def check_budget(self, extra_virtual=0.0):
        if self.closed:
            raise BudgetError("会话已经结束。")
        if (self.deadline is not None
                and time.monotonic() >= self.deadline - self.exit_reserve):
            raise BudgetError("现实时间不足，预留退出时间。")
        if self.virtual_time + extra_virtual >= self.max_virtual - 1.0:
            raise BudgetError("接近虚拟时间上限。")

    # 内部统一请求入口，/enter、/measure、/clear、/exit 都通过这里发送
    def _call(self, path, position=None, channel=None):
        if self.closed:
            raise ProtocolError("不能向已结束的会话发送动作。")
        if path not in ("/enter", "/exit"):
            if (len(position) != 2
                    or any(not math.isfinite(x) or abs(x) > 2000000
                           for x in position)):
                raise ValueError("坐标必须有限且在协议范围内。")
            if type(channel) is not int or not 1 <= channel <= 20:
                raise ValueError("频道必须是1至20的整数。")
            self.check_budget(distance(self.position, position) / 5 + 6)
        self.counter += 1
        payload = dict(arena_id="default", robot_id=self.robot_id,
                       request_id="%s-%s-%d" % (self.prefix,
                                                path[1:], self.counter))
        if position is not None:
            payload.update(position={"x": position[0], "y": position[1]},
                           channel=channel)
        deadline = self.deadline
        if deadline is not None and path != "/exit":
            deadline -= self.exit_reserve
        before = time.monotonic()
        try:
            response = self.transport.request(path, payload, deadline)
        except Exception as e:
            self._log(dict(path=path, request=payload, error=str(e)))
            raise
        self._log(dict(path=path, request=payload, response=response))
        if response.get("accepted") is not True:
            raise ProtocolError(
                "%s 被拒绝；检查参赛队号、接口状态及请求字段。" % path)
        vt = response.get("virtual_time_s")
        if (not isinstance(vt, (int, float)) or not math.isfinite(vt)
                or vt < self.virtual_time - 1e-5):
            raise ProtocolError("虚拟时钟异常。")
        if path == "/enter":
            left = response.get("remaining_real_duration_s")
            if (not isinstance(left, (int, float))
                    or not math.isfinite(left) or left < 0):
                raise ProtocolError("/enter 缺少有效剩余时间。")
            self.started_at = before
            self.deadline = before + left
            self.max_virtual = float(
                response.get("max_virtual_duration_s",
                             max_virtual_seconds_limit))
        elif position is not None:
            self.metrics["distance_m"] += distance(self.position, position)
            self.position = tuple(position)
            if path == "/measure":
                kind = response.get("measure_result")
                if kind not in ("direction", "near", "no_signal"):
                    raise ProtocolError("未知检测结果。")
                if kind == "direction":
                    deg = response.get("svd_deg")
                    if (not isinstance(deg, (int, float))
                            or not math.isfinite(deg)
                            or not 0 <= deg < 360):
                        raise ProtocolError("示向度缺失或越界。")
                self.metrics["measurements"] += 1
                self.metrics["channel_switches"] += int(
                    channel != self.channel)
                self.channel = channel
            else:
                kind = response.get("clear_result")
                if kind not in ("success", "no_target_in_range"):
                    raise ProtocolError("未知清除结果。")
                self.metrics["clear_attempts"] += 1
                self.metrics["cleared"] += int(kind == "success")
                self.metrics["failed_clears"] += int(kind != "success")
        self.virtual_time = float(vt)
        if path == "/exit":
            self.closed = True
        return response

    # 把请求和响应写入 jsonl 日志
    def _log(self, record):
        if self.log:
            record["local_monotonic_s"] = time.monotonic()
            self.log.write(json.dumps(record, ensure_ascii=False,
                                      allow_nan=False) + "\n")
            self.log.flush()

    # 进入模拟器
    def enter(self):
        return self._call("/enter")

    # 在指定位置、指定频道测量
    def measure(self, position, channel):
        return self._call("/measure", position, channel)

    # 在指定位置、指定频道尝试清除
    def clear(self, position, channel):
        return self._call("/clear", position, channel)

    # 退出模拟器
    def exit(self):
        return self._call("/exit")

    # 关闭日志文件
    def close_log(self):
        if self.log:
            self.log.close()


# ========== 多层环绕扫描点 ==========
# 多层同心圆上的探测点，覆盖内部和外部（对外朝外的定向源关键）
def generate_layered_probes():
    points = [(0.0, 0.0)]
    # 内层到边缘
    for r, n in [(400.0, 8), (800.0, 12), (1200.0, 16),
                 (1600.0, 20), (1850.0, 22)]:
        for i in range(n):
            a = full_circle * i / n + 0.03
            points.append((r * math.cos(a), r * math.sin(a)))
    # 外层（竞技场外）
    for r, n in [(2000.0, 24), (2200.0, 24)]:
        for i in range(n):
            a = full_circle * i / n + 0.03
            points.append((r * math.cos(a), r * math.sin(a)))
    return points


# ========== 路径规划 ==========
# 最近邻 + 2-opt 优化。nodes 元素格式 (tag, ident, position)
def open_route(start, nodes):
    if len(nodes) < 2:
        return list(nodes)
    remaining = list(nodes)
    route, p = [], start
    while remaining:
        node = min(remaining, key=lambda t: distance(p, t[2]))
        remaining.remove(node)
        route.append(node)
        p = node[2]
    for _ in range(5):
        improved = False
        for i in range(len(route) - 1):
            a = start if i == 0 else route[i - 1][2]
            for j in range(i + 1, len(route)):
                b, c = route[i][2], route[j][2]
                old, new = distance(a, b), distance(a, c)
                if j + 1 < len(route):
                    d = route[j + 1][2]
                    old += distance(c, d)
                    new += distance(b, d)
                if new + 1e-6 < old:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
        if not improved:
            break
    return route


# ========== 主求解器 ==========
# 主求解器，负责初始扫描、目标定位、未知频道扫描、清除和缺失判定
class Solver:
    def __init__(self, client, optimized=True, verbose=True, **kwargs):
        self.client = client
        self.tracks = {ch: Track(ch) for ch in all_channels}
        self.optimized = optimized
        self.verbose = verbose
        self.complete = False
        self.completion_reason = "incomplete"
        self.localize_attempts = {}
        # 分层探测点
        self.probes = generate_layered_probes()
        # 每个频道已经尝试过的探测点索引（用于避免重复）
        self.probe_cursor = {ch: 0 for ch in all_channels}
        # 每个探测点对应能覆盖的频道集合
        self.probe_pos = [p for p in self.probes]

    # 按需打印进度信息
    def say(self, message):
        if self.verbose:
            print(message, flush=True)

    # 返回所有正在追踪的频道
    def tracked(self):
        return [t for t in self.tracks.values() if t.status == "tracked"]

    # 返回所有仍未知的频道
    def unknown(self):
        return [t for t in self.tracks.values() if t.status == "unknown"]

    # 返回所有暂时卡住的频道
    def stuck(self):
        return [t for t in self.tracks.values() if t.status == "stuck"]

    # 统计已经发现或已清除的频道数量
    def detected_count(self):
        return sum(t.status in ("tracked", "cleared")
                   for t in self.tracks.values())

    # ---------- 探测点与已用点判断 ----------
    # 返回某个频道已经探测过的所有位置
    def _tried_points(self, track):
        pts = [q for q, _ in track.observations]
        pts += [q for q, _ in track.no_signal_history]
        return pts

    # 给某频道找一个尚未探测过且距离已用点较远的探测点
    def _next_fresh_probe(self, track):
        tried = self._tried_points(track)
        # 自适应最小间距
        n = len(tried)
        if n > 260:
            min_d = 25.0
        elif n > 140:
            min_d = 40.0
        elif n > 60:
            min_d = 55.0
        else:
            min_d = 70.0

        best, best_score = None, -1.0
        for i, p in enumerate(self.probe_pos):
            # 与已用点的最小距离
            if tried:
                md = min(distance(p, q) for q in tried)
                if md < min_d:
                    continue
            # 与当前位置的距离（近的更优）
            cur = distance(self.client.position, p)
            score = (min(md, 900.0) if tried else 900.0) - cur * 0.5
            if score > best_score:
                best_score = score
                best = p
        return best

    # 严格的多方向覆盖判据
    def _is_confidently_absent(self, track):
        if track.observations:
            return False
        hist = track.no_signal_history
        if len(hist) < absent_min_no_signal_count:
            return False
        # 角度扇区覆盖
        angle_bins = set()
        for (x, y), _ in hist:
            a = math.atan2(y, x) % full_circle
            angle_bins.add(int(a / full_circle * 12))
        if len(angle_bins) < absent_min_angle_sector_count:
            return False
        # 径向覆盖
        radius_bins = set()
        for (x, y), _ in hist:
            r = math.hypot(x, y)
            radius_bins.add(min(int(r / 500), 4))
        if len(radius_bins) < absent_min_radius_band_count:
            return False
        return True

    # ---------- 单点观测 ----------
    # 在指定位置、指定频道执行一次测量，并更新对应 Track
    def observe(self, position, channel):
        track = self.tracks[channel]
        response = self.client.measure(position, channel)
        kind = response["measure_result"]

        if kind == "no_signal":
            track.no_signal_history.append((tuple(position), None))
            return kind

        if track.status == "cleared":
            raise ModelError("已清除频道重新出现信号")

        if track.status == "unknown":
            track.status = "tracked"
            track.poly = outer_disk((0.0, 0.0), field_radius)
        elif track.status == "stuck":
            track.status = "tracked"
            if not track.poly:
                track.poly = outer_disk((0.0, 0.0), field_radius)
            self.localize_attempts[track.channel] = 0

        if self.localize_attempts.get(track.channel, 0) >= max_localize_try_count:
            self.localize_attempts[track.channel] = 0

        if kind == "near":
            track.near_point = tuple(position)
            track.observations.append((tuple(position), None))
        else:
            track.near_point = None
            angle = response["svd_deg"]
            new_poly = clip_bearing(track.poly, position, angle)
            if new_poly:
                new_poly = clip_disk_outer(new_poly, position, max_measure_range)
                if new_poly:
                    track.poly = new_poly
            track.observations.append((tuple(position), angle))
        return kind

    # ---------- 可靠清除 ----------
    # 当可行域半径足够小时，执行清除；如果失败，会在附近多个偏移点重试
    def _clear_certified(self, track, center, radius):
        if radius > safe_clear_radius:
            raise ModelError("无法保证清除（半径 %.2f 米）" % radius)
        result = self.client.clear(center, track.channel)
        if result["clear_result"] == "success":
            track.status = "cleared"
            self.say("已清除频道 %d（累计 %d）" % (
                track.channel, self.client.metrics["cleared"]))
            return
        # 邻近重试
        for off in [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0),
                    (6.0, 0.0), (-6.0, 0.0), (0.0, 6.0), (0.0, -6.0),
                    (10.0, 0.0), (-10.0, 0.0), (0.0, 10.0), (0.0, -10.0)]:
            p = (center[0] + off[0], center[1] + off[1])
            result = self.client.clear(p, track.channel)
            if result["clear_result"] == "success":
                track.status = "cleared"
                self.say("已清除频道 %d（邻域，累计 %d）" % (
                    track.channel, self.client.metrics["cleared"]))
                return
        track.status = "stuck"
        self.say("频道 %d 清除失败，暂时放弃" % track.channel)

    # ---------- 目标定位的探测点选择 ----------
    # 对被追踪目标，选择下一个测量点（优先方向、次优先环绕）
    def _choose_position(self, track):
        center, radius = track.circle()
        tried = self._tried_points(track)
        n = len(tried)
        min_d = 25.0 if n > 260 else (40.0 if n > 140 else
                                      (55.0 if n > 60 else 65.0))

        def is_new(p):
            return all(distance(p, q) >= min_d for q in tried)

        def clip(p):
            r = math.hypot(p[0], p[1])
            limit = field_radius + outside_probe_margin
            if r > limit:
                s = limit / r
                return (p[0] * s, p[1] * s)
            return p

        dir_obs = [(p, a) for p, a in track.observations if a is not None]

        # 策略 A：多边形较大时，绕多边形中心环绕
        if radius > 260.0 and dir_obs:
            for scale in (0.45, 0.65, 0.28, 0.85, 0.15, 1.0):
                for k in range(10):
                    a = full_circle * k / 10 + 0.07
                    p = clip((center[0] + radius * scale * math.cos(a),
                              center[1] + radius * scale * math.sin(a)))
                    if is_new(p):
                        return p

        # 策略 B：沿最近示向度方向前进
        if dir_obs:
            last_pos, last_angle = dir_obs[-1]
            a = math.radians(last_angle)
            dx, dy = math.cos(a), math.sin(a)
            perp = (-dy, dx)
            if radius > 1100:
                step = 550.0
            elif radius > 550:
                step = 400.0
            elif radius > 220:
                step = 250.0
            else:
                step = 140.0
            for d in (step, step * 1.4, step * 0.7,
                      240.0, 500.0, 800.0, 1100.0):
                for h in (0.0, 0.30 * d, -0.30 * d,
                          0.60 * d, -0.60 * d,
                          0.90 * d, -0.90 * d):
                    p = clip((last_pos[0] + d * dx + h * perp[0],
                              last_pos[1] + d * dy + h * perp[1]))
                    if is_new(p):
                        return p

        # 策略 C：围绕多边形中心做小环绕
        if radius < field_radius * 1.2:
            for scale in (0.5, 0.8, 0.3, 1.0, 0.15, 1.4):
                for k in range(12):
                    a = full_circle * k / 12
                    p = clip((center[0] + radius * scale * math.cos(a),
                              center[1] + radius * scale * math.sin(a)))
                    if is_new(p):
                        return p

        # 策略 D：竞技场（含外侧）随机点
        for _ in range(320):
            a = random.uniform(0.0, full_circle)
            r = random.uniform(0.0, field_radius + outside_probe_margin)
            p = clip((r * math.cos(a), r * math.sin(a)))
            if is_new(p):
                return p

        # 兜底
        a = random.uniform(0.0, full_circle)
        r = random.uniform(200.0, field_radius - 100)
        return (r * math.cos(a), r * math.sin(a))

    # ---------- 单步定位 ----------
    # 对某个已发现目标执行一步定位；如果可行域足够小，就直接清除
    def _localize_step(self, track):
        center, radius = track.circle()
        if radius <= safe_clear_radius:
            self._clear_certified(track, center, radius)
            return
        count = self.localize_attempts.get(track.channel, 0)
        if count >= max_localize_try_count:
            track.status = "stuck"
            self.say("频道 %d 定位超过 %d 次尝试，暂时放弃"
                     % (track.channel, max_localize_try_count))
            return
        self.localize_attempts[track.channel] = count + 1
        p = self._choose_position(track)
        kind = self.observe(p, track.channel)
        if kind == "near":
            self._clear_certified(track, tuple(p), 2.0)

    # ---------- 未知频道探测任务 ----------
    # 为所有 unknown/stuck 频道规划一个探测点，返回 (position, channels)
    def _next_scan_task(self):
        # 候选频道：unknown / stuck / 定位次数超限的 tracked
        pending = [t for t in self.tracks.values()
                   if t.status in ("unknown", "stuck")
                   or (t.status == "tracked"
                       and self.localize_attempts.get(t.channel, 0)
                       >= max_localize_try_count)]
        if not pending:
            return None

        # 为每个 pending 频道找它下一个 fresh probe
        per_channel = {}
        for t in pending:
            p = self._next_fresh_probe(t)
            if p is not None:
                per_channel.setdefault(p, []).append(t.channel)

        if not per_channel:
            # 所有点都用尽 → 随机
            a = random.uniform(0.0, full_circle)
            r = random.uniform(200.0, field_radius)
            p = (r * math.cos(a), r * math.sin(a))
            return p, [t.channel for t in pending]

        # 贪心：优先能覆盖多个频道的探测点，且离当前位置近
        cur = self.client.position
        best_p, best_score, best_chs = None, -1.0, []
        for p, chs in per_channel.items():
            score = len(chs) / (1.0 + distance(cur, p) / 400.0)
            if score > best_score:
                best_score, best_p, best_chs = score, p, chs

        return best_p, best_chs

    # ---------- 主循环 ----------
    # 主求解流程：初始扫描 → 主循环 → 完成判据
    def solve(self):
        # ========== 第 1 步：初始扫描 ==========
        self.say("=== 初始扫描 ===")
        for t in list(self.tracks.values()):
            try:
                kind = self.observe((0.0, 0.0), t.channel)
            except ModelError:
                continue
            if kind == "near":
                self._clear_certified(t, (0.0, 0.0), 2.0)
        self.say("初始：清除 %d | 追踪 %s" % (
            self.client.metrics["cleared"],
            [t.channel for t in self.tracked()]))

        # ========== 第 2 步：主循环 ==========
        for round_num in range(max_main_loop_rounds):
            self.client.check_budget()

            # 虚拟时间紧张时收缩行为
            time_left = self.client.max_virtual - self.client.virtual_time
            if time_left < 2000:
                self.say("虚拟时间接近上限，停止搜索")
                break

            cleared = self.client.metrics["cleared"]
            if cleared >= max_source_count:
                break

            # 优先级 1：清除任何可行域半径足够小的目标
            done = False
            for t in self.tracked():
                c, r = t.circle()
                if r <= safe_clear_radius:
                    self._clear_certified(t, c, r)
                    done = True
                    break
            if done:
                continue

            # 优先级 2：定位被追踪的频道
            tracked = self.tracked()
            fresh = [t for t in tracked
                     if self.localize_attempts.get(t.channel, 0)
                     < max_localize_try_count]
            if fresh:
                def prio(t):
                    c, r = t.circle()
                    return r + distance(self.client.position, c) / 500.0
                t = min(fresh, key=prio)
                self._localize_step(t)
                continue

            # 优先级 3：扫描未知 / 卡住频道
            task = self._next_scan_task()
            if task is None:
                break
            p, chs = task

            if round_num % 10 == 0:
                self.say("轮次 %d：清除 %d | 探测 (%.0f, %.0f) | 频道 %s"
                         % (round_num, cleared, p[0], p[1], chs))

            # 距离太远时，用中间过渡点（降低单次移动成本）
            cur = self.client.position
            if distance(cur, p) > 2600.0:
                # 直连（不做过路点，避免更差）
                pass

            for ch in chs:
                if self.detected_count() >= max_source_count:
                    break
                t = self.tracks[ch]
                # 每次都重新判定（可能中途已被标记为 absent）
                if t.status not in ("unknown", "stuck", "tracked"):
                    continue
                try:
                    kind = self.observe(p, ch)
                except ModelError:
                    continue
                if kind == "near":
                    self._clear_certified(t, p, 2.0)

            # 严格 absent 判据
            for t in self.tracks.values():
                if t.status == "unknown" and self._is_confidently_absent(t):
                    t.status = "absent"
                    self.say("频道 %d 多方向无信号，标记不存在（%d 次）"
                             % (t.channel, len(t.no_signal_history)))

        # ========== 第 3 步：完成判据 ==========
        cleared = self.client.metrics["cleared"]
        if cleared >= max_source_count:
            self.complete = True
            self.completion_reason = "cleared_16_upper_bound"
        elif not self.tracked() and not self.unknown() and not self.stuck():
            self.complete = True
            self.completion_reason = (
                "resolved_all_channels_cleared_%d" % cleared)
        else:
            self.complete = False
            self.completion_reason = (
                "best_effort_cleared_%d_of_%d" % (cleared, max_source_count))

        self.say("=== 结束：清除 %d | 追踪 %s | 未知 %s | 卡住 %s | absent %s ==="
                 % (cleared,
                    [t.channel for t in self.tracked()],
                    [t.channel for t in self.unknown()],
                    [t.channel for t in self.stuck()],
                    [t.channel for t in self.tracks.values()
                     if t.status == "absent"]))
        return self.summary()

    # 汇总统计信息
    def summary(self):
        m = dict(self.client.metrics)
        m.update(
            complete=self.complete,
            completion_reason=self.completion_reason,
            total_virtual_s=self.client.virtual_time,
            average_time_s=(self.client.virtual_time / m["cleared"]
                            if m["cleared"] else None),
            local_elapsed_s=(time.monotonic() - self.client.started_at
                             if self.client.started_at is not None else 0),
            tracked_channels=[t.channel for t in self.tracked()],
            unknown_channels=[t.channel for t in self.unknown()],
            stuck_channels=[t.channel for t in self.stuck()],
            coverage_absent_channels=[
                t.channel for t in self.tracks.values()
                if t.status == "absent"])
        return m


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-id", default="", help="参赛队号，运行时通过命令传入")
    parser.add_argument("--base-url", default=server_base_url)
    parser.add_argument("--timeout", type=float, default=default_timeout_seconds)
    parser.add_argument("--log-dir", default="b3_logs")
    parser.add_argument("--baseline", action="store_true",
                        help="兼容旧参数，本版本忽略")
    args = parser.parse_args()

    if args.robot_id in ("你的参赛队号", "", "<参赛队号>"):
        parser.error(
            "请修改顶部 team_robot_id，或传入 --robot-id 你的实际参赛队号")
    if args.timeout <= 0:
        parser.error("--timeout 必须为正数")

    directory = Path(args.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("b3_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    log_path = directory / (stem + ".jsonl")
    client = Client(HttpTransport(args.base_url, args.timeout),
                    args.robot_id, log_path)
    solver = Solver(client, optimized=True)

    exit_ok, error = False, None
    try:
        print("请确认模拟器已进入问题4演练/正式测试，且接口已就绪。",
              flush=True)
        response = client.enter()
        print("进入成功，本次现实时间预算：%s s"
              % response["remaining_real_duration_s"], flush=True)
        solver.solve()
        client.exit()
        exit_ok = True
    except (KeyboardInterrupt, BudgetError, ModelError) as e:
        error = str(e) or "用户中断"
        if (client.deadline is not None
                and time.monotonic() < client.deadline
                and not client.closed):
            try:
                client.exit()
                exit_ok = True
            except (TransportError, ProtocolError, BudgetError) as ex:
                error += "; 退出未确认: " + str(ex)
    except (TransportError, ProtocolError) as e:
        error = str(e)
    finally:
        summary = solver.summary()
        summary.update(
            exit_confirmed=exit_ok,
            error=error,
            strategy="layered_probe_v2",
            statistics_source="HTTP responses; source total not exposed")
        summary_path = directory / (stem + "_summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        client.close_log()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("本地记录：", log_path)
    return 0 if solver.complete and exit_ok and not error else 1


if __name__ == "__main__":
    raise SystemExit(main())