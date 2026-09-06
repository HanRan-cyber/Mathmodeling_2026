import numpy as np
from scipy.optimize import minimize

# ==================== 常量 ====================
G = 9.8
V_MISSILE = 300.0
R_CLOUD = 10.0
CLOUD_DURATION = 20.0
SINK_RATE = 3.0
V_MIN, V_MAX = 70.0, 140.0

M1_0 = np.array([20000.0, 0.0, 2000.0])
FY1_0 = np.array([17800.0, 0.0, 1800.0])
TARGET = np.array([0.0, 200.0, 0.0])          # 真目标（下底面圆心）
dir_missile = -M1_0 / np.linalg.norm(M1_0)   # 导弹飞行单位方向（指向假目标原点）

def missile_pos(t):
    """导弹在 t 时刻的位置"""
    return M1_0 + V_MISSILE * dir_missile * t

def cloud_center(burst_point, t, burst_time):
    if t < burst_time or t > burst_time + CLOUD_DURATION:
        return None
    return burst_point - np.array([0.0, 0.0, SINK_RATE * (t - burst_time)])

def is_effective(cloud_center, m_pos, t_pos):
    """判断是否有效遮蔽（距离≤10且投影在线段内）"""
    vec = t_pos - m_pos
    w = cloud_center - m_pos
    proj = np.dot(w, vec) / np.dot(vec, vec)
    if proj < 0 or proj > 1:
        return False
    dist = np.linalg.norm(np.cross(w, vec)) / np.linalg.norm(vec)
    return dist <= R_CLOUD

def compute_shielding_duration(params):
    """
    输入决策变量 [v, theta, t_drop, delay]
    返回有效遮蔽时长（秒）
    """
    v, theta, t_drop, delay = params
    if not (V_MIN <= v <= V_MAX and t_drop >= 0 and delay >= 0):
        return 0.0

    burst_time = t_drop + delay
    # 投放点（高度恒为1800）
    drop_x = FY1_0[0] + v * t_drop * np.cos(theta)
    drop_y = FY1_0[1] + v * t_drop * np.sin(theta)
    drop_point = np.array([drop_x, drop_y, 1800.0])

    # 起爆点（水平匀速，垂直自由落体）
    bx = drop_x + v * delay * np.cos(theta)
    by = drop_y + v * delay * np.sin(theta)
    bz = 1800.0 - 0.5 * G * delay**2
    if bz < 0:
        return 0.0
    burst_point = np.array([bx, by, bz])

    total = 0.0
    dt = 0.01
    t_start = burst_time
    t_end = burst_time + CLOUD_DURATION
    for t in np.arange(t_start, t_end, dt):
        m_pos = missile_pos(t)
        if np.linalg.norm(m_pos - TARGET) < 1.0:
            break
        center = cloud_center(burst_point, t, burst_time)
        if center is None:
            continue
        if is_effective(center, m_pos, TARGET):
            total += dt
    return total

# ==================== 枚举可行初值 ====================
best_duration = 0.0
best_params = None
feasible_params_list = []   # 保存所有可行参数（即使时长=0）

print("开始枚举候选起爆点...")
for t_b in np.arange(0.0, 61.0, 0.2):          # 起爆时刻步长0.2s
    m_b = missile_pos(t_b)
    for s in np.arange(0.05, 1.0, 0.05):       # 位置比例步长0.05
        B = m_b + s * (TARGET - m_b)           # 理想起爆点（起爆瞬时云团中心）
        # 检查无人机能否到达该起爆点
        dx = B[0] - FY1_0[0]
        dy = B[1] - FY1_0[1]
        horiz_dist = np.sqrt(dx**2 + dy**2)
        dz = 1800.0 - B[2]
        if dz < 0:
            continue
        delay = np.sqrt(2 * dz / G) if dz > 0 else 0.0
        if t_b < delay:
            continue
        if t_b == 0:
            continue
        v = horiz_dist / t_b
        if v < V_MIN or v > V_MAX:
            continue
        theta = np.arctan2(dy, dx)
        t_drop = t_b - delay
        if t_drop < 0:
            continue

        params = [v, theta, t_drop, delay]
        # 计算实际遮蔽时长（可能为0）
        dur = compute_shielding_duration(params)
        feasible_params_list.append(params)
        if dur > best_duration:
            best_duration = dur
            best_params = params
            print(f"新最优: v={v:.1f}, θ={np.degrees(theta):.1f}°, t_drop={t_drop:.1f}s, delay={delay:.1f}s, 时长={dur:.3f}s")

# 如果没有找到任何可行解，则报错
if not feasible_params_list:
    raise RuntimeError("未找到任何可行参数，请检查模型或放宽约束。")

# 若best_params为None（所有可行参数时长均为0），则取第一个可行参数作为初值
if best_params is None:
    best_params = feasible_params_list[0]
    print(f"警告：所有可行参数遮蔽时长为0，取第一个作为初值：v={best_params[0]:.1f}, θ={np.degrees(best_params[1]):.1f}°")

print(f"\n枚举结束，共找到 {len(feasible_params_list)} 组可行参数。最佳时长 = {best_duration:.3f}s")

# ==================== 局部精化 ====================
def objective(params):
    return -compute_shielding_duration(params)

bounds = [(V_MIN, V_MAX), (0.0, 2*np.pi), (0.0, 60.0), (0.0, 20.0)]
res = minimize(objective, best_params, method='SLSQP',
               bounds=bounds, options={'ftol': 1e-6, 'maxiter': 300})
opt_params = res.x
opt_duration = -res.fun

# ==================== 输出最终结果 ====================
v_opt, theta_opt, t_drop_opt, delay_opt = opt_params
burst_time_opt = t_drop_opt + delay_opt
drop_x = FY1_0[0] + v_opt * t_drop_opt * np.cos(theta_opt)
drop_y = FY1_0[1] + v_opt * t_drop_opt * np.sin(theta_opt)
drop_point = np.array([drop_x, drop_y, 1800.0])
bx = drop_x + v_opt * delay_opt * np.cos(theta_opt)
by = drop_y + v_opt * delay_opt * np.sin(theta_opt)
bz = 1800.0 - 0.5 * G * delay_opt**2
burst_point = np.array([bx, by, bz])

print("\n===== 问题2 最终最优策略 =====")
print(f"无人机速度      : {v_opt:.2f} m/s")
print(f"航向角          : {np.degrees(theta_opt):.2f}°")
print(f"投放时刻        : {t_drop_opt:.2f} s")
print(f"引信延时        : {delay_opt:.2f} s")
print(f"起爆时刻        : {burst_time_opt:.2f} s")
print(f"投放点坐标      : ({drop_point[0]:.1f}, {drop_point[1]:.1f}, {drop_point[2]:.1f})")
print(f"起爆点坐标      : ({burst_point[0]:.1f}, {burst_point[1]:.1f}, {burst_point[2]:.1f})")
print(f"最大有效遮蔽时长: {opt_duration:.3f} s")