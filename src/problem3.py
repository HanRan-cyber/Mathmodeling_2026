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

import numpy as np


# ========== 参数设置 ==========
# 模拟器地址与参赛队号
base_url = "http://127.0.0.1:2026"
robot_id = "" #空字符串，运行时通过--robot-id传入

# 题目给定的物理常量
arena_radius = 1800.0              # 目标区域半径（米）
min_range = 1000.0                 # 干扰源最小有效接收半径
max_range = 1500.0                 # 干扰源最大有效接收半径
clear_radius = 20.0                # 光学精确定位距离
safe_clear_radius = 19.95          # 清除前留一点数值余量
angle_error = math.radians(1.006)  # ±1°环境误差 + 0.005°舍入 + 数值余量

channels = range(1, 21)            # 20 个频道编号
max_sources = 16                   # 干扰源数量上限
tau = 2.0 * math.pi                # tau 就是 2π，也就是整个圆的弧度

# 保底搜索点：原点 + 半径 anchor_radius 的圆周上每 60° 一个
anchor_radius = 1200.0
num_anchors = 6

# 覆盖网格参数
coverage_sectors = 72  # sector是区域的意思，72是适中的选择方案
coverage_radii = [0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0]

# HTTP 通信
default_timeout = 4.0
default_retries = 2
max_virtual_seconds = 360000.0     # 虚拟时间上限（100 小时）
exit_reserve_seconds = 3.0         # 退出前预留的现实时间

# 定位与调度
side_ratio = 0.15                  # 定位时横向偏移比例，和problem2采取的策略不同，以数量取胜
max_localize_iter = 32             # 单目标定位最大迭代次数
max_schedule_round = 100           # 全局调度最大轮数，防止任务陷入死循环

# 扫描门限
scan_min_gain = 96                 # 普通清除点新增覆盖至少这么多三角形才顺便扫
cross_threshold = 0.30             # 顺路交会的最小 sin 阈值，约17度，大于17度值得停下来测量


# ========== 基础工具函数 ==========
def distance(a, b):  # 两点直线距离
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ========== 异常类 ==========
class ModelError(RuntimeError):
    pass


class ProtocolError(RuntimeError):
    pass


class TransportError(RuntimeError):
    pass


class BudgetError(RuntimeError):
    pass


# ========== 几何剪裁 ==========
def clip_halfplane(poly, nx, ny, c):
    """凸多边形与半平面 nx*x + ny*y <= c 求交，边界视为内部。"""
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

#造一个包住圆的多边形,顶点半径略大于 radius
def outer_disk(center, radius, n=72):
    rv = radius / math.cos(math.pi / n)
    angles = tau * np.arange(n) / n
    xs = center[0] + rv * np.cos(angles)
    ys = center[1] + rv * np.sin(angles)
    return list(zip(xs.tolist(), ys.tolist()))

#保留圆内的部分,即凸多边形与圆盘求交
def clip_disk_outer(poly, center, radius, n=32):
    for i in range(n):
        a = tau * (i + 0.5) / n
        nx, ny = math.cos(a), math.sin(a)
        poly = clip_halfplane(poly, nx, ny,
                              radius + nx * center[0] + ny * center[1])
    return poly

#保留方向范围内的部分，把凸多边形与 ±angle_error 的示向度楔形求交
def clip_bearing(poly, position, degrees):
    a = math.radians(degrees)
    lo, hi = a - angle_error, a + angle_error
    # 下边界：射线左侧；上边界：射线右侧
    for nx, ny in ((math.sin(lo), -math.cos(lo)),
                   (-math.sin(hi), math.cos(hi))):
        poly = clip_halfplane(poly, nx, ny,
                              nx * position[0] + ny * position[1])
    return poly


# ========== 最小包围圆 ==========
#一两个点为直径画圆
def _diameter_circle(a, b):
    c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return c, distance(a, b) / 2

#过三个点画圆
def _three_circle(a, b, c):
    # 平移到 a，减小大坐标下的抵消误差
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

#主函数，计算出最小包围圆的圆心和半径
def enclosing_circle(poly):
    if not poly:
        raise ModelError("定位可行域为空，不能继续声称位置可靠。")
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


# ========== 证明是否覆盖 ==========
# 把 1800m 圆域切成 792 个小三角形。
# 如果某次 no_signal，且某个三角形三个顶点都落在
# (当前点, 1000m) 圆内，就把这个三角形从该频道候选区域中排除。
# 圆是凸集，所以这是对连续区域的保守证明，不是点采样。
class Coverage:
    def __init__(self):
        n = coverage_sectors
        r_outer = arena_radius / math.cos(math.pi / n)
        radii = coverage_radii + [r_outer]
        self.cells = []
        for i in range(n):
            a, b = tau * i / n, tau * (i + 1) / n
            for r0, r1 in zip(radii, radii[1:]):
                p = (r0 * math.cos(a), r0 * math.sin(a))
                q = (r1 * math.cos(a), r1 * math.sin(a))
                r = (r1 * math.cos(b), r1 * math.sin(b))
                self.cells.append((p, q, r))
                if r0:
                    s = (r0 * math.cos(b), r0 * math.sin(b))
                    self.cells.append((p, r, s))
        # 另存一个 numpy 数组，形状 (n_cells, 3, 2)，
        # 用于快速向量化地计算覆盖掩码。
        self.cells_arr = np.asarray(self.cells, dtype=float)
        self.full = (1 << len(self.cells)) - 1
        self.anchors = [(0.0, 0.0)] + [
            (anchor_radius * math.cos(tau * i / num_anchors),
             anchor_radius * math.sin(tau * i / num_anchors))
            for i in range(num_anchors)]
        self._cache = {}
        self.anchor_masks = [self.mask(p) for p in self.anchors]
        # 检查七个保底点是否真的把整个区域都覆盖住了
        union = 0
        for mask in self.anchor_masks:
            union |= mask
        if union != self.full:
            raise ModelError("七点保底扫描没有完整覆盖，请检查覆盖网格参数。")

    def mask(self, position):
        # 用 numpy 向量化：一次性算出所有三角形三个顶点到 position 的距离平方
        position = tuple(position)
        if position not in self._cache:
            pos = np.asarray(position, dtype=float)           # (2,)
            diff = self.cells_arr - pos                       # (n_cells, 3, 2)
            d2 = np.sum(diff * diff, axis=2)                  # (n_cells, 3)
            inside = np.all(d2 <= (min_range - 1e-6) ** 2, axis=1)  # (n_cells,)
            mask = 0
            for i in np.flatnonzero(inside):
                mask |= 1 << int(i)
            self._cache[position] = mask
        return self._cache[position]


# ========== 单频道状态 ==========
# 每个频道最多一个干扰源。状态：
#   unknown — 还没在这个频道收到任何信号
#   tracked — 已收到信号，正在定位
#   cleared — 已成功清除
#   absent  — 通过覆盖证书排除了这个频道
class Track:
    def __init__(self, channel, remaining):
        self.channel = channel
        self.remaining = remaining
        self.status = "unknown"
        self.poly = []              # 当前可行的凸多边形
        self.observations = []      # [(观测点, 示向度), ...]
        self.near_point = None      # 如果测到 near，直接记住位置

    def circle(self):
        # 如果测到过 near，位置就直接用那个点，半径 5 米
        if self.near_point is not None:
            return self.near_point, 5.0
        return enclosing_circle(self.poly)


# ========== HTTP 传输 ==========
# 串行 HTTP。只有传输失败才重试，重试时请求体和 request_id 完全一致，
# 保证服务器不会重复收费。
class HttpTransport:
    def __init__(self, base_url, timeout=default_timeout,
                 retries=default_retries):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

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
                # 有 HTTP 错误响应时终止，不伪造 accepted=false 的当前状态
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
                    delay = max(0, min(delay, deadline - time.monotonic()))
                time.sleep(delay)
                continue
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError) as e:
                raise ProtocolError(
                    "响应不是有效JSON；动作结果不确定，停止发送新动作。") from e
            if status != 200 or not isinstance(result, dict):
                raise ProtocolError("HTTP状态或响应结构异常。")
            return result
        raise TransportError(
            "连接失败或响应丢失，停止发送新动作：%s" % last)


# ========== 客户端：与模拟器通信 ==========
class Client:
    def __init__(self, transport, robot_id, log_path=None):
        self.transport = transport
        self.robot_id = robot_id
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_time = 0.0
        self.max_virtual = max_virtual_seconds
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

    def check_budget(self, extra_virtual=0.0):
        if self.closed:
            raise BudgetError("会话已经结束。")
        if (self.deadline is not None
                and time.monotonic() >= self.deadline - self.exit_reserve):
            raise BudgetError("现实时间不足，预留退出时间。")
        if self.virtual_time + extra_virtual >= self.max_virtual - 1.0:
            raise BudgetError("接近虚拟时间上限。")

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
            raise ProtocolError("虚拟时钟异常；保留最后确认状态并停止。")
        if path == "/enter":
            left = response.get("remaining_real_duration_s")
            if (not isinstance(left, (int, float))
                    or not math.isfinite(left) or left < 0):
                raise ProtocolError(
                    "/enter 缺少有效 remaining_real_duration_s。")
            self.started_at = before
            # 从请求开始计时，保守计入首个请求的网络往返时间
            self.deadline = before + left
            self.max_virtual = float(
                response.get("max_virtual_duration_s",
                             max_virtual_seconds))
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
                # /clear 不更新测向机频道
        self.virtual_time = float(vt)
        if path == "/exit":
            self.closed = True
        return response

    def _log(self, record):
        if self.log:
            record["local_monotonic_s"] = time.monotonic()
            self.log.write(json.dumps(record, ensure_ascii=False,
                                      allow_nan=False) + "\n")
            self.log.flush()

    def enter(self):
        return self._call("/enter")

    def measure(self, position, channel):
        return self._call("/measure", position, channel)

    def clear(self, position, channel):
        return self._call("/clear", position, channel)

    def exit(self):
        return self._call("/exit")

    def close_log(self):
        if self.log:
            self.log.close()


# ========== 路线规划 ==========
#共两个任务：去清理已发现的目标；保底搜索点，去扫描未知频道
#策略：就近先走最近路线：遇到交叉就转向；走完就停，不会原地
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
    for _ in range(6):
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
class Solver:
    def __init__(self, client, optimized=True, verbose=True,
                 side_ratio=side_ratio,
                 scan_min_gain=scan_min_gain,
                 cross_threshold=cross_threshold):
        self.client = client
        self.coverage = Coverage()
        self.tracks = {ch: Track(ch, self.coverage.full)
                       for ch in channels}
        self.optimized = optimized
        self.verbose = verbose
        self.side_ratio = side_ratio
        self.scan_min_gain = scan_min_gain
        self.cross_threshold = cross_threshold
        self.complete = False
        self.completion_reason = "incomplete"

    def say(self, message):
        if self.verbose:
            print(message, flush=True)

    def tracked(self):
        return [t for t in self.tracks.values() if t.status == "tracked"]

    def unknown(self):
        return [t for t in self.tracks.values() if t.status == "unknown"]

    def detected_count(self):
        return sum(t.status in ("tracked", "cleared")
                   for t in self.tracks.values())

    # ---------- 单点观测 ----------
    def observe(self, position, channel):
        track = self.tracks[channel]
        response = self.client.measure(position, channel)
        kind = response["measure_result"]
        if kind == "no_signal":
            if track.status == "unknown":
                track.remaining &= ~self.coverage.mask(position)
                if not track.remaining:
                    track.status = "absent"
            # 已知目标的无信号信息是非凸圆外约束；本算法保守忽略它，
            # 不因一次无信号删除目标，也不把 1500m 当成保底接收距离。
            return kind
        if track.status in ("cleared", "absent"):
            raise ModelError("已清除或已排除的频道重新出现信号，"
                             "请核对题型和规则。")
        if track.status == "unknown":
            track.status = "tracked"
            track.poly = outer_disk((0.0, 0.0), arena_radius)
        if kind == "near":
            track.near_point = tuple(position)
        else:
            angle = response["svd_deg"]
            poly = clip_bearing(track.poly, position, angle)
            poly = clip_disk_outer(poly, position, max_range)
            if not poly:
                raise ModelError(
                    "频道%d的±1.006°定位约束互相矛盾。" % channel)
            track.poly = poly
            track.observations.append((tuple(position), angle))
        return kind

    # ---------- 同一地点批量扫描 ----------
    def scan(self, position, opportunistic=True, force_unknown=True):
        mask = self.coverage.mask(position)
        # 优先测当前频道，减少频道切换
        todo = [t.channel for t in self.unknown()
                if (t.remaining & mask) and
                (force_unknown
                 or (t.remaining & mask).bit_count() >= self.scan_min_gain
                 or (t.remaining & ~mask) == 0)]
        todo.sort(key=lambda ch: (ch != self.client.channel, ch))
        for ch in todo:
            if self.detected_count() >= max_sources:
                break
            self.observe(position, ch)
        # 借清除/搜索点为其他目标交会，不额外走路
        if opportunistic and self.optimized:
            for t in list(self.tracked()):
                if t.near_point is not None or not t.observations:
                    continue
                center, radius = t.circle()
                if (radius <= safe_clear_radius
                        or distance(position, center) > 1000):
                    continue
                if any(distance(position, p) < 1
                       for p, _ in t.observations):
                    continue
                last = t.observations[-1][0]
                ax, ay = center[0] - last[0], center[1] - last[1]
                bx, by = center[0] - position[0], center[1] - position[1]
                den = math.hypot(ax, ay) * math.hypot(bx, by)
                sin_cross = abs(ax * by - ay * bx) / max(den, 1e-9)
                if sin_cross >= self.cross_threshold:
                    self.observe(position, t.channel)

    # ---------- 可靠清除 ----------
    def _clear_certified(self, track, center, radius):
        # 包围圆以内的所有候选点都必须落在 20m 清除圆内
        p = center
        d = distance(self.client.position, center)
        slack = max(0.0, safe_clear_radius - radius)
        if d > 1e-8:
            f = min(slack, d) / d
            p = (center[0] + f * (self.client.position[0] - center[0]),
                 center[1] + f * (self.client.position[1] - center[1]))
        result = self.client.clear(p, track.channel)
        if result["clear_result"] != "success":
            raise ModelError(
                "频道%d的保证清除点失败，停止而不丢弃目标。"
                % track.channel)
        track.status = "cleared"
        self.say("已清除 %2d 个 | 频道 %2d | 虚拟时间 %.1f s" % (
            self.client.metrics["cleared"], track.channel,
            self.client.virtual_time))

    # ---------- 单目标定位 ----------
    def localize(self, channel):
        track = self.tracks[channel]
        for _ in range(max_localize_iter):
            self.client.check_budget()
            center, radius = track.circle()
            if radius <= safe_clear_radius:
                self._clear_certified(track, center, radius)
                return
            # 站在可行域中部，偏到长轴一侧，兼顾交会角与路程
            a = max(track.poly, key=lambda p: distance(p, center))
            ux, uy = a[0] - center[0], a[1] - center[1]
            norm = max(math.hypot(ux, uy), 1e-8)
            nx, ny = -uy / norm, ux / norm
            offset = min(180.0, max(8.0, self.side_ratio * radius))
            offset = min(offset, max(0.0, 990.0 - radius))
            if radius >= 985:
                raise ModelError(
                    "定位范围异常大，无法保证下一点收到全向信号。")
            candidates = [(center[0] + sign * offset * nx,
                           center[1] + sign * offset * ny)
                          for sign in (-1, 1)]
            p = min(candidates,
                    key=lambda q: distance(self.client.position, q))
            if any(distance(p, old) < 0.5
                   for old, _ in track.observations):
                p = (candidates[1] if p == candidates[0]
                     else candidates[0])
            kind = self.observe(p, channel)
            if kind == "no_signal":
                raise ModelError(
                    "保证接收点未收到信号；"
                    "此策略仅适用于问题3全向模型。")
        raise ModelError("单目标定位迭代超过 %d 次，保留日志供检查。"
                         % max_localize_iter)

    # ---------- 选择下一个任务点 ----------
    def next_task(self):
        known = self.tracked()
        nodes = [("target", t.channel, t.circle()[0]) for t in known]
        remaining = 0
        if self.detected_count() < max_sources:
            for t in self.unknown():
                remaining |= t.remaining
        # 待清除点预计也会顺便扫描，先用它们安排覆盖
        for _, _, p in nodes:
            remaining &= ~self.coverage.mask(p)
        choices = list(enumerate(self.coverage.anchors))
        while remaining:
            idx, p = max(
                choices,
                key=lambda item: (
                    (self.coverage.anchor_masks[item[0]]
                     & remaining).bit_count()
                    / (400 + distance(self.client.position, item[1]))))
            covered = self.coverage.anchor_masks[idx] & remaining
            if not covered:
                raise ModelError("存在未覆盖区域，但没有有效保底点。")
            nodes.append(("scan", idx, p))
            remaining &= ~covered
            choices.remove((idx, p))
        if not nodes:
            return None
        return open_route(self.client.position, nodes)[0]

    # ---------- 主循环 ----------
    def solve(self):
        # ========== 第 1 步：在原点先粗扫一遍 ==========
        self.scan((0.0, 0.0), opportunistic=False)

        if self.optimized:
            # ========== 第 2 步：循环调度 ==========
            # 每轮从待清除目标和保底搜索点里，用最近邻 + 2-opt
            # 找下一个任务点
            for _ in range(max_schedule_round):
                self.client.check_budget()
                task = self.next_task()
                if task is None:
                    break
                kind, ident, position = task
                if kind == "target":
                    # 目标点：交会定位 → 清除 → 顺便扫未知频道
                    self.localize(ident)
                    self.scan(self.client.position,
                              force_unknown=False)
                else:
                    # 搜索点：只做扫描
                    self.scan(position)
            else:
                raise ModelError("全局调度超过 %d 轮，未确认完成。"
                                 % max_schedule_round)
        else:
            # ========== 参考策略：固定七点 ==========
            for p in self.coverage.anchors:
                self.scan(p, opportunistic=False)
                while self.tracked():
                    track = min(
                        self.tracked(),
                        key=lambda t: distance(self.client.position,
                                           t.circle()[0]))
                    self.localize(track.channel)
                if self.client.metrics["cleared"] == max_sources:
                    break

        # ========== 第 3 步：判定是否真正完成 ==========
        # 只有下面两种情况才算完成，不能清到 10 个就停。
        cleared = self.client.metrics["cleared"]
        if cleared == max_sources:
            self.complete = True
            self.completion_reason = "cleared_16_upper_bound"
        elif not self.tracked() and not self.unknown() and cleared >= 10:
            self.complete = True
            self.completion_reason = (
                "all_channels_cleared_or_coverage_certified_absent")
        if not self.complete:
            raise ModelError("未满足完整清除判据，不报告成功。")
        return self.summary()

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
    parser.add_argument("--base-url", default=base_url)
    parser.add_argument("--timeout", type=float,
                        default=default_timeout)
    parser.add_argument("--log-dir", default="b3_logs")
    parser.add_argument("--baseline", action="store_true",
                        help="仅供演练比较的固定七点参考策略")
    args = parser.parse_args()

    if args.robot_id in ("你的参赛队号", "", "<参赛队号>"):
        parser.error(
            "请修改顶部 robot_id，或传入 --robot-id 你的实际参赛队号")
    if args.timeout <= 0:
        parser.error("--timeout 必须为正数")

    directory = Path(args.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("b3_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    log_path = directory / (stem + ".jsonl")
    client = Client(HttpTransport(args.base_url, args.timeout),
                    args.robot_id, log_path)
    solver = Solver(client, optimized=not args.baseline)

    exit_ok, error = False, None
    try:
        print("请确认模拟器已进入问题3演练/正式测试，"
              "且接口已经就绪。", flush=True)
        response = client.enter()
        print("进入成功，本次现实时间预算：%s s"
              % response["remaining_real_duration_s"], flush=True)
        solver.solve()
        client.exit()
        exit_ok = True
    except (KeyboardInterrupt, BudgetError, ModelError) as e:
        error = str(e) or "用户中断"
        # 已知本地停止、会话尚在有效期内时主动退出
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
            strategy=("fixed_reference" if args.baseline
                      else "adaptive_coverage_2opt"),
            statistics_source=("HTTP responses; source total "
                               "not exposed by protocol"))
        summary_path = directory / (stem + "_summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        client.close_log()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("本地记录：", log_path)
        print("正式案例编码和正式程序运行时间请以模拟器为准；"
              "正式加密日志需在模拟器中导出。")
    return 0 if solver.complete and exit_ok and not error else 1


if __name__ == "__main__":
    raise SystemExit(main())