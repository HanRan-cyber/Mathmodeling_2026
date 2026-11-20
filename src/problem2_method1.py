import random
import numpy as np
import matplotlib.pyplot as plt

# ========== 参数设置 ==========
radius = 1800  # radius是目标区域半径，题目给定1800m
error = 1.0  # error是示向度误差，题目给定为1度
min_receive_radius = 1000  # 最小有效接收半径
max_receive_radius = 1500  # 最大有效接收半径
estimate_distance = 1250  # 估计干扰源距离时用的参考距离，取中间值
max_distance_to_estimated_point = 1000  # 第二个检测点到估计点P的最大距离
min_baseline = 200  # 第一个检测点与第二个检测点的最小间距
min_distance_to_estimated_point = 50  # 新增：第二个检测点到估计点P的最小保护距离
target_angle = 90.0  # 理想交会角
angle_low = 60.0  # 交会角可接受下限
angle_high = 120.0  # 交会角可接受上限
penalty = 1e4  # 惩罚系数，调大以适应新fitness的量级


# ========== 基础工具函数 ==========
def distance(point1, point2):  # 用于计算两点之间的距离
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


def angle_between(vector1, vector2):  # 用于计算两个向量之间的夹角
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ========== 几何计算 ==========
def estimate_source(first_point, measured_bearing):  #估计点P的位置
    bearing_rad = np.radians(measured_bearing)  # 角度转弧度
    P_x = first_point[0] + estimate_distance * np.cos(bearing_rad)
    P_y = first_point[1] + estimate_distance * np.sin(bearing_rad)
    return np.array([P_x, P_y])

#用于计算交会角的大小
def cross_angle(first_point, second_point, estimated_point):
    location_first_point = np.array([first_point[0], first_point[1]])
    location_second_point = np.array([second_point[0], second_point[1]])
    location_estimated_point = np.array([estimated_point[0], estimated_point[1]])
    return angle_between(location_first_point - location_estimated_point,
                         location_second_point - location_estimated_point)


# ========== 约束与评价函数 ==========
def in_area(point):  # 判断点是否在规定范围内
    point = np.array(point)
    if np.sum(point ** 2) > radius ** 2:
        return False
    else:
        return True

#对S2对位置进行约束
def feasible(second_point, first_point, estimated_point):
    # 1. 在目标区域内
    if not in_area(second_point):
        return False
    # 2. 到估计点P的距离约束：不能太远（收不到信号），也不能太近（交会几何退化）
    distance_second_point_to_estimated_point = distance(second_point, estimated_point)
    if distance_second_point_to_estimated_point > max_distance_to_estimated_point:
        return False
    if distance_second_point_to_estimated_point < min_distance_to_estimated_point:
        return False
    # 3. 与第一个检测点的距离不能太近
    distance_second_point_to_first_point = distance(first_point, second_point)
    if distance_second_point_to_first_point < min_baseline:
        return False

    return True


# 评价第二个检测点的得分，越小越好，基于重叠面积
def fitness(second_point, first_point, estimated_point):
    gamma = cross_angle(first_point, second_point, estimated_point)  # 交会角(度)
    sin_gamma = np.sin(np.radians(gamma))

    # 防止 sin_gamma 接近 0 时数值爆炸
    if sin_gamma < 1e-3:
        sin_gamma = 1e-3

    d2 = distance(second_point, estimated_point)  # S2 到估计点 P 的距离
    return d2 / sin_gamma


# 带惩罚的评价函数，用于粒子群
def penalty_eval(second_point, first_point, estimated_point):
    base_fitness = fitness(second_point, first_point, estimated_point)
    if feasible(second_point, first_point, estimated_point):
        return base_fitness
    return base_fitness + penalty  # 不可行时加上惩罚系数


# ========== 网格搜索 ==========
def grid_search(first_point, estimated_point, step=50.0, fitness_threshold=None):
    best_second_point = None
    best_value = float('inf')  #初始化，使best_value变的无穷大
    candidates = []
    #candidates是满足约束且适应度较好的点列表
    x = -radius
    while x <= radius:
        y = -radius
        while y <= radius:
            second_point = (x, y)
            if feasible(second_point, first_point, estimated_point):
                value = fitness(second_point, first_point, estimated_point)
                if value < best_value:
                    best_value = value
                    best_second_point = second_point
                candidates.append((second_point, value))
            y += step
        x += step

    # 筛选出适应度不超过最优值 1.2 倍的点
    if best_value < float('inf'):
        if fitness_threshold is None:
            fitness_threshold = 1.2 * best_value
        candidates = [p for p, v in candidates if v <= fitness_threshold]

    return best_second_point, best_value, candidates


# ========== 粒子群搜索（过程式） ==========
def pso_search(first_point, estimated_point, num_particles=40, num_iter=120):
    particles = []

    # 1. 初始化粒子
    for i in range(num_particles):
        # 采样：在估计点周围随机取可行点
        while True:
            r = max_distance_to_estimated_point * np.sqrt(random.random())
            angle = random.uniform(0, 2 * np.pi)
            x = estimated_point[0] + r * np.cos(angle)
            y = estimated_point[1] + r * np.sin(angle)
            if in_area((x, y)) and distance((x, y), first_point) >= min_baseline:
                break

        vx = random.uniform(-50, 50)
        vy = random.uniform(-50, 50)
        value = penalty_eval((x, y), first_point, estimated_point)

        particles.append({
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'p_best_x': x, 'p_best_y': y,
            'p_best_value': value
        })

    # 2. 找初始全局最优
    g_best = min(particles, key=lambda p: p['p_best_value'])
    g_best_x = g_best['p_best_x']
    g_best_y = g_best['p_best_y']
    g_best_value = g_best['p_best_value']

    # 3. 迭代
    for t in range(num_iter):
        w = 0.9 - 0.5 * t / num_iter  # 惯性权重递减
        c1 = 1.5
        c2 = 1.5

        for p in particles:
            r1 = random.random()
            r2 = random.random()

            # 标准粒子群速度更新
            p['vx'] = w * p['vx'] + c1 * r1 * (p['p_best_x'] - p['x']) + c2 * r2 * (g_best_x - p['x'])
            p['vy'] = w * p['vy'] + c1 * r1 * (p['p_best_y'] - p['y']) + c2 * r2 * (g_best_y - p['y'])

            # 位置更新
            p['x'] += p['vx']
            p['y'] += p['vy']

            # 边界保护：如果粒子飞出目标区域，强行拉回圆内
            if not in_area((p['x'], p['y'])):
                r = np.hypot(p['x'], p['y'])
                p['x'] = p['x'] / r * radius
                p['y'] = p['y'] / r * radius

            # 计算新适应度，更新p_best和g_best
            value = penalty_eval((p['x'], p['y']), first_point, estimated_point)

            if value < p['p_best_value']:
                p['p_best_value'] = value
                p['p_best_x'] = p['x']
                p['p_best_y'] = p['y']

                if value < g_best_value:
                    g_best_value = value
                    g_best_x = p['x']
                    g_best_y = p['y']

    return (g_best_x, g_best_y), g_best_value


# ========== 主流程及绘图 ==========
def plot_result(first_point, measured_bearing, estimated_point,
                best_second_point, candidates=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 目标区域圆
    circle = plt.Circle((0, 0), radius, fill=False,
                        color='gray', linewidth=1.5, linestyle='--',
                        label=f'Target area (R={radius} m)', zorder=1)
    ax.add_patch(circle)

    # 2. 候选区域（散点，半透明）
    if candidates:
        cx = [p[0] for p in candidates]
        cy = [p[1] for p in candidates]
        ax.scatter(cx, cy, s=8, c='orange', alpha=0.4,
                   label='Candidate region', zorder=2)

    # 3. 示向度射线
    bearing_rad = np.radians(measured_bearing)
    L = 1800
    ax.plot([first_point[0], first_point[0] + L * np.cos(bearing_rad)],
            [first_point[1], first_point[1] + L * np.sin(bearing_rad)],
            'r--', linewidth=1.5, label='Bearing from S1', zorder=3)

    # 4. 连线 S1-P 和 S2-P（画在点下面）
    if best_second_point is not None:
        ax.plot([first_point[0], estimated_point[0]],
                [first_point[1], estimated_point[1]],
                'g-', linewidth=1.5, zorder=3)
        ax.plot([best_second_point[0], estimated_point[0]],
                [best_second_point[1], estimated_point[1]],
                'b-', linewidth=1.5, zorder=3)

    # 5. 绘制 S1
    ax.plot(first_point[0], first_point[1], 'ro', markersize=12,
            markeredgecolor='white', markeredgewidth=1.5,
            label='S1 (first detector)', zorder=4)

    # 6. 绘制 S2
    if best_second_point is not None:
        ax.plot(best_second_point[0], best_second_point[1], 'bs', markersize=12,
                markeredgecolor='white', markeredgewidth=1.5,
                label='S2 (optimal second detector)', zorder=5)

    # 7. 最后绘制 P，确保永远在最上层，不会被 S2 挡住
    ax.plot(estimated_point[0], estimated_point[1], 'g^', markersize=14,
            markeredgecolor='white', markeredgewidth=1.5,
            label='P (estimated source)', zorder=6)

    # 8. 在 P 处标注交会角
    if best_second_point is not None:
        gamma = cross_angle(first_point, best_second_point, estimated_point)
        v1 = np.array(first_point) - np.array(estimated_point)
        v2 = np.array(best_second_point) - np.array(estimated_point)
        v1 = v1 / (np.linalg.norm(v1) + 1e-9)
        v2 = v2 / (np.linalg.norm(v2) + 1e-9)
        bisector = v1 + v2
        bisector = bisector / (np.linalg.norm(bisector) + 1e-9)
        text_x = estimated_point[0] + 200 * bisector[0]
        text_y = estimated_point[1] + 200 * bisector[1]
        ax.text(text_x, text_y, f'γ = {gamma:.1f}°',
                fontsize=13, color='purple', fontweight='bold', zorder=7)

    ax.set_aspect('equal')
    ax.set_xlim(-radius * 1.1, radius * 1.1)
    ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Problem 2: Selection of the Second Detection Point')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('problem2_result.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    # 固定示例
    first_point = (0.0, 0.0)
    measured_bearing = 45.0

    estimated_point = estimate_source(first_point, measured_bearing)
    print("估计干扰源位置 =", estimated_point)

    # 网格搜索粗找
    grid_second_point, grid_value, candidates = grid_search(
        first_point, estimated_point, step=100.0)

    # PSO 细找
    pso_second_point, pso_value = pso_search(
        first_point, estimated_point, num_particles=40, num_iter=120)

    print("PSO 第二个检测点 =", pso_second_point, " 评价值 =", pso_value)
    if pso_second_point is not None:
        print("交会角 =", cross_angle(first_point, pso_second_point, estimated_point))

    # 画图
    plot_result(first_point, measured_bearing, estimated_point,
                pso_second_point, candidates)