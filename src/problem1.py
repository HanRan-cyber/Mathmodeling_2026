import math
import random

# ========== 参数设置 ==========
radius = 1800  #radius是半径的意思，题目给定1800m

num_detectors = 5  #num_detectors代表生成检测点的个数，这里取5个

error = 1.0  #error代表误差，题目给定为1度

# ========== 随机生成干扰源真实位置 ==========
# 在半径为radius 的圆内随机生成一个点
angle_source = random.uniform(0, 2 * math.pi)  # 先随机一个角度（0 到 2π）
r_source = radius * math.sqrt(random.uniform(0, 1))  # 再随机一个半径，用 sqrt 是为了让点在圆内均匀分布
# 转换成直角坐标
source_x = r_source * math.cos(angle_source)
source_y = r_source * math.sin(angle_source)

# ========== 生成检测点并计算示向度 ==========
# 用来保存所有检测点的信息
detectors = []

for i in range(num_detectors):
    # 在目标区域内随机生成检测点坐标
    angle_detector = random.uniform(0, 2 * math.pi)
    r_detector = radius * math.sqrt(random.uniform(0, 1))
    detector_x = r_detector * math.cos(angle_detector)
    detector_y = r_detector * math.sin(angle_detector)

    # 计算从检测点到干扰源的真实方位角,即干扰点向量 - 检测点向量
    true_x = source_x - detector_x
    true_y = source_y - detector_y

    # atan2 返回弧度，范围 (-π, π]
    true_rad = math.atan2(true_y, true_x)
    # 转换成角度
    true_degree = math.degrees(true_rad)
    # 调整到 [0, 360) 范围
    if true_degree < 0:
        true_degree += 360.0

    # 加入随机误差，误差在 [-max_error_deg, max_error_deg] 之间
    random_error = random.uniform(-error, error)
    measured_degree = true_degree + random_error
    measured_degree = measured_degree % 360.0  # 保证示向度也在 [0, 360) 范围内

    # 保存这个检测点的信息
    detectors.append({
        'x': detector_x,
        'y': detector_y,
        'true_bearing': true_degree,
        'measured_bearing': measured_degree
    })

    # # 打印出来，方便查看
    # print(f"检测点 {i+1}：")
    # print(f"  坐标 = ({det_x:.2f}, {det_y:.2f}) 米")
    # print(f"  真实方位角 = {true_degree:.2f} 度")
    # print(f"  测得示向度 = {measured_degree:.2f} 度")
    # print()

# ========== 把角域变成平面约束 ==========

# 对每个检测点，根据测得示向度，计算两条边界射线的方向向量
# 下边界 = 示向度 - 1°，上边界 = 示向度 + 1°
for i, det in enumerate(detectors):
    # 取该检测点测得的示向度
    measured = det['measured_bearing']

    # 下边界角度（度）
    lower_angle_degree = measured - error
    # 上边界角度（度）
    upper_angle_degree = measured + error

    # 转成弧度
    lower_angle_rad = math.radians(lower_angle_degree)
    upper_angle_rad = math.radians(upper_angle_degree)

    # 下边界方向向量
    lower_direction_x = math.cos(lower_angle_rad)
    lower_direction_y = math.sin(lower_angle_rad)

    # 上边界方向向量
    upper_dir_x = math.cos(upper_angle_rad)
    upper_dir_y = math.sin(upper_angle_rad)

    # 把这些信息存进 det 里，方便后面用
    det['lower_direction'] = (lower_direction_x, lower_direction_y)
    det['upper_direction'] = (upper_dir_x, upper_dir_y)



# ========== 计算定位区域的凸多边形顶点 ==========

def cross(a, b):
    """二维向量叉积：a × b"""
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(p1, d1, p2, d2):
    """
    求两条直线的交点。
    直线1：经过点 p1，方向向量 d1
    直线2：经过点 p2，方向向量 d2
    返回交点 (x, y)，如果平行则返回 None。
    """
    # p1 + t * d1 = p2 + s * d2
    # 写成叉积形式：t = cross(p2 - p1, d2) / cross(d1, d2)
    p1 = (p1[0], p1[1])
    p2 = (p2[0], p2[1])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    denom = cross(d1, d2)
    if abs(denom) < 1e-12:
        return None  # 两条直线平行，没有交点

    t = (dx * d2[1] - dy * d2[0]) / denom
    x = p1[0] + t * d1[0]
    y = p1[1] + t * d1[1]
    return (x, y)


def is_in_angle_range(point, det):
    """判断点 point 是否在检测点 det 的角域内"""
    vx = point[0] - det['x']
    vy = point[1] - det['y']

    ld = det['lower_direction']
    ud = det['upper_direction']

    cross_lower = ld[0] * vy - ld[1] * vx
    cross_upper = ud[0] * vy - ud[1] * vx

    return cross_lower >= -1e-9 and cross_upper <= 1e-9


def is_in_all_angle_ranges(point, detectors):
    """判断点是否在所有检测点的角域交集内"""
    for det in detectors:
        if not is_in_angle_range(point, det):
            return False
    return True


# ========== 收集所有边界线 ==========
# 每条边界线用 (检测点坐标, 方向向量) 表示
boundary_lines = []

for det in detectors:
    p = (det['x'], det['y'])
    boundary_lines.append((p, det['lower_direction']))
    boundary_lines.append((p, det['upper_direction']))


# ========== 两两求交点，筛选出满足所有约束的点 ==========
vertices = []  #vertice是顶点的意思

n_lines = len(boundary_lines)
for i in range(n_lines):
    for j in range(i + 1, n_lines):
        p1, d1 = boundary_lines[i]
        p2, d2 = boundary_lines[j]

        point = line_intersection(p1, d1, p2, d2)
        if point is None:
            continue

        # 检查这个交点是否在目标区域内
        if point[0] ** 2 + point[1] ** 2 > radius ** 2:
            continue

        # 检查是否满足所有角域约束
        if is_in_all_angle_ranges(point, detectors):
            vertices.append(point)


# ========== 去重 ==========
unique_vertices = []
for p in vertices:
    is_duplicate = False
    for q in unique_vertices:
        if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 1e-6:
            is_duplicate = True
            break
    if not is_duplicate:
        unique_vertices.append(p)


# ========== 按极角排序，形成凸多边形 ==========
if len(unique_vertices) >= 3:
    # 求这些点的重心，作为排序中心
    cx = sum(p[0] for p in unique_vertices) / len(unique_vertices)
    cy = sum(p[1] for p in unique_vertices) / len(unique_vertices)

    # 按相对于重心的极角排序
    unique_vertices.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


# ========== 输出中间数据，方便检查 ==========
print("========== 基础数据 ==========")
print(f"干扰源真实位置：({source_x:.4f}, {source_y:.4f})")
print(f"目标区域半径：{radius} 米")
print(f"检测点个数：{num_detectors}")
print()

print("========== 各检测点信息 ==========")
for i, det in enumerate(detectors):
    print(f"检测点 {i+1}：")
    print(f"  坐标       = ({det['x']:.4f}, {det['y']:.4f})")
    print(f"  真实方位角 = {det['true_bearing']:.4f}°")
    print(f"  测得示向度 = {det['measured_bearing']:.4f}°")
print()

print("========== 定位区域顶点 ==========")
if len(unique_vertices) < 3:
    print(f"顶点数 = {len(unique_vertices)}，少于 3 个，无法构成多边形。")
    print("可能原因：角域没有公共交集，或检测点布设导致交会失败。")
    print("程序结束。")
    raise SystemExit
else:
    for i, p in enumerate(unique_vertices):
        print(f"  顶点 {i+1}：({p[0]:.4f}, {p[1]:.4f})")
    print(f"共 {len(unique_vertices)} 个顶点，已按极角排序。")
print()


# ========== 计算凸多边形的直径 ==========

# 用来保存当前找到的最远两点
best_pair = [unique_vertices[0], unique_vertices[0]]
# 当前最大距离
best_distance = 0.0

# 双重循环枚举所有顶点对
for i in unique_vertices:
    for j in unique_vertices:
        dx = i[0] - j[0]
        dy = i[1] - j[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 如果发现更远的点对，就更新
        if distance > best_distance:
            best_distance = distance
            best_pair[0] = i
            best_pair[1] = j

print("========== 多边形直径 ==========")
print(f"最远两点 A = ({best_pair[0][0]:.4f}, {best_pair[0][1]:.4f})")
print(f"最远两点 B = ({best_pair[1][0]:.4f}, {best_pair[1][1]:.4f})")
print(f"直径 D = {best_distance:.4f} 米")
print()


# ========== 判断以直径为直径的圆能否覆盖整个凸多边形 ==========

# 圆心：直径两端点的中点
center_x = (best_pair[0][0] + best_pair[1][0]) / 2
center_y = (best_pair[0][1] + best_pair[1][1]) / 2

# 半径：直径的一半
R = best_distance / 2

print("========== 直径圆信息 ==========")
print(f"圆心：({center_x:.4f}, {center_y:.4f})")
print(f"半径：{R:.4f} 米")
print()

# 逐个顶点检查是否在圆内
print("========== 顶点覆盖检查 ==========")
can_cover = True
max_vertex_distance = 0.0  # 记录所有顶点中离圆心最远的距离，便于分析
for idx, p in enumerate(unique_vertices):
    d = math.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2)
    if d > max_vertex_distance:
        max_vertex_distance = d
    status = "在圆内" if d <= R + 1e-9 else "在圆外"
    print(f"  顶点 {idx+1} ({p[0]:.4f}, {p[1]:.4f})：到圆心距离 = {d:.4f}，{status}")
    if d > R + 1e-9:
        can_cover = False

print()
print(f"顶点到圆心的最大距离 = {max_vertex_distance:.4f} 米")
print(f"圆半径               = {R:.4f} 米")
print()

print("========== 最终结论 ==========")
if can_cover:
    print("以该直径为直径的圆可以覆盖整个定位区域（凸多边形）。")
else:
    print("以该直径为直径的圆不能覆盖整个定位区域（凸多边形）。")
    print("原因：存在顶点到圆心的距离大于半径 R。")


# ========== 附加验证：真实干扰源的位置情况 ==========
print()
print("========== 真实干扰源验证 ==========")
source_point = (source_x, source_y)

# 1) 真实源是否在定位区域内
if is_in_all_angle_ranges(source_point, detectors):
    print("真实干扰源位于定位区域内（符合预期）。")
else:
    print("真实干扰源不在定位区域内（异常，请检查误差设置或角域方向）。")

# 2) 真实源到直径圆圆心的距离
d_source = math.sqrt((source_x - center_x) ** 2 + (source_y - center_y) ** 2)
if d_source <= R + 1e-9:
    print(f"真实干扰源到圆心距离 = {d_source:.4f}，在直径圆内。")
else:
    print(f"真实干扰源到圆心距离 = {d_source:.4f}，不在直径圆内。")




