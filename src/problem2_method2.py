import os
import time
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 参数设置 ==========
radius = 1500.0                # 扇形半径，单位：米
half_degree = 1.0              # 扇形半张角，单位：度
wedge_half_degree = 1.0        # 光束半宽，单位：度
cell = 1.0                     # 网格边长，单位：米
x_min, x_max = 700.0, 1500.0   # 第二个检测点候选区域 x 范围
y_min, y_max = 0.0, 1500.0     # 第二个检测点候选区域 y 范围
num_theta = 200                # 数值积分时把 ±1° 分成 200 段
chunk_size = 4000              # 批量计算时一次处理多少个方向
eps = 1e-9                     # 浮点误差容差
use_parallel = True            # 是否启用多进程并行


# ========== 基础工具函数 ==========
#把扇形区域描成一个多边形，用于画图
def sector_polygon(n_arc=400):
    theta = np.linspace(np.radians(-half_degree),
                        np.radians(half_degree), n_arc)
    arc = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return np.vstack([np.zeros((1, 2)), arc, np.zeros((1, 2))])

#直接计算扇形面积，用于校验
def sector_area():
    return (2 * half_degree) / 360.0 * np.pi * radius ** 2

#在扇形区域内按 1 米 × 1 米网格取格子中心点
def sample_sector_points():
    x_centers = np.arange(cell / 2, radius, cell)
    half_span = radius * np.tan(np.radians(half_degree))
    y_centers = np.arange(-half_span, half_span, cell)
    X, Y = np.meshgrid(x_centers, y_centers)
    r = np.hypot(X, Y)
    ang = np.degrees(np.arctan2(Y, X))
    mask = (r <= radius) & (np.abs(ang) <= half_degree)
    return np.column_stack([X[mask], Y[mask]])

#在第二个检测点候选区域内按 1 米 × 1 米网格取格子中心点
def s2_candidates():
    xs = np.arange(x_min + cell / 2, x_max, cell)
    ys = np.arange(y_min + cell / 2, y_max, cell)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


# ========== 核心数学：计算一束光与扇形交集的面积 ==========
#计算交集的面积
def batch_wedge_area(s2, alphas):
    out = np.empty(alphas.shape[0])
    for i in range(0, len(alphas), chunk_size):
        out[i:i + chunk_size] = _batch_wedge_area_one_chunk(
            s2, alphas[i:i + chunk_size])
    return out

#单个批次的面积计算
def _batch_wedge_area_one_chunk(s2, alphas):
    a1 = np.radians(alphas - wedge_half_degree)[:, None]
    a2 = np.radians(alphas + wedge_half_degree)[:, None]
    c1, s1 = np.cos(a1), np.sin(a1)
    c2, s2c = np.cos(a2), np.sin(a2)

    th = np.linspace(-half_degree, half_degree, num_theta) * np.pi / 180
    ct, st = np.cos(th), np.sin(th)

    B1 = c1 * st - s1 * ct
    B2 = c2 * st - s2c * ct
    A1 = s1 * s2[0] - c1 * s2[1]
    A2 = s2c * s2[0] - c2 * s2[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = -A1 / B1
        r2 = -A2 / B2

    lo = np.maximum(np.zeros_like(B1),
                    np.maximum(np.where(B1 > 0, r1, -np.inf),
                               np.where(B2 < 0, r2, -np.inf)))
    hi = np.where(B1 < 0, np.minimum(np.full_like(B1, radius), r1), radius)
    hi = np.where(B2 > 0, np.minimum(hi, r2), hi)

    lo = np.clip(lo, 0.0, radius)
    hi = np.clip(hi, 0.0, radius)
    seg = np.maximum((hi * hi - lo * lo) / 2.0, 0.0)

    bad = ((np.abs(B1) < 1e-12) & (A1 < -1e-9)) | \
          ((np.abs(B2) < 1e-12) & (A2 > 1e-9))
    seg = np.where(bad, 0.0, seg)

    return np.trapezoid(seg, th)

#计算一个 S2 点的累加面积
def eval_s2(s2, pts1):
    v = pts1 - s2
    d = np.hypot(v[:, 0], v[:, 1])
    keep = d > eps
    if not keep.any():
        return (float(s2[0]), float(s2[1]), 0.0)
    alphas = np.degrees(np.arctan2(v[keep, 1], v[keep, 0]))
    areas = batch_wedge_area(s2, alphas)
    return (float(s2[0]), float(s2[1]), float(areas.sum()))


# ========== 多分辨率搜索 ==========
#生成一批待测试的 S2 点
def grid_points(center, step, half=None):
    if half is None:
        xs = np.arange(x_min + cell / 2, x_max, step)
        ys = np.arange(y_min + cell / 2, y_max, step)
    else:
        x0 = np.round((center[0] - half - cell / 2) / step) * step + cell / 2
        y0 = np.round((center[1] - half - cell / 2) / step) * step + cell / 2
        xs = np.arange(x0, center[0] + half + cell / 2 + 1e-9, step)
        ys = np.arange(y0, center[1] + half + cell / 2 + 1e-9, step)
        xs = xs[(xs >= x_min + cell / 2) & (xs <= x_max - cell / 2)]
        ys = ys[(ys >= y_min + cell / 2) & (ys <= y_max - cell / 2)]
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])

#多进程调用需要的包装函数
def eval_s2_wrapper(args):
    s2, pts1 = args
    return eval_s2(s2, pts1)

#对一批 S2 点计算累加面积，返回最优结果和全部结果
def run_grid(points, pts1, workers):
    if use_parallel and workers > 1 and len(points) > 8:
        with Pool(workers) as pool:
            res = pool.map(eval_s2_wrapper, [(p, pts1) for p in points])
    else:
        res = [eval_s2(p, pts1) for p in points]
    arr = np.asarray(res)
    i = int(np.argmin(arr[:, 2]))
    return arr[i], arr


# ========== 绘图 ==========
#画出扇形、候选区域方框和最优 S2 点
def draw_result(sector, best):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.fill(sector[:, 0], sector[:, 1], color="steelblue",
            alpha=0.3, label="扇形区域")
    ax.plot(sector[:, 0], sector[:, 1], color="steelblue", linewidth=1.2)
    ax.plot([0, radius], [0, 0], "r--", linewidth=1.2)
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            "g--", linewidth=1.5,
            label=f"候选区域 x∈({x_min:g},{x_max:g}), y∈({y_min:g},{y_max:g})")
    ax.plot(best[0], best[1], "o", ms=12, color="#d64541", zorder=5)
    ax.annotate(f"S2=({best[0]:.1f}, {best[1]:.1f})\nA={best[2]:,.0f} m²",
                xy=(best[0], best[1]), xytext=(best[0] - 260, best[1] + 220),
                fontsize=11, fontweight="bold", color="#d64541",
                arrowprops=dict(arrowstyle="->", color="#d64541"))
    ax.plot(0, 0, "ko", ms=7)
    ax.annotate("S1 (0,0)", xy=(0, 0), xytext=(60, -130), fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("累加面积最小的 S2 点位置")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-250, 1750)
    ax.set_ylim(-300, 1700)
    plt.tight_layout()
    plt.savefig("strategy2_result.png", dpi=150)
    plt.close(fig)


# ========== 主流程 ==========
def main():
    t_start = time.perf_counter()

    print("[第 1 步] 建立扇形区域 ...")
    sector = sector_polygon()
    area = sector_area()
    print(f"    S1=(0,0)，x 轴为中心线，±{half_degree}°，"
          f"R={radius} m，扇形面积={area:.1f} m²")

    print("[第 2 步] 扇形内按 1m×1m 网格取点 P ...")
    pts1 = sample_sector_points()
    print(f"    第一段点 P 数量: {len(pts1)}，密度 "
          f"{len(pts1) / area:.4f} 点/m²")

    print("[第 3 步] 第二个检测点候选区域 ...")
    s2_all = s2_candidates()
    print(f"    全量候选 S2: {len(s2_all)} 个，"
          f"区域面积 {(x_max - x_min) * (y_max - y_min):.0f} m²")

    print("[第 4/5 步] 多分辨率搜索累加面积最小的 S2 ...")
    workers = max(1, os.cpu_count() or 1)
    print(f"    并行核数: {workers}")

    # 粗筛
    coarse = grid_points(None, 50)
    best, _ = run_grid(coarse, pts1, workers)
    print(f"    [粗筛 50m] {len(coarse)} 点 → "
          f"S2=({best[0]:.1f}, {best[1]:.1f}) A={best[2]:.3f} m²")

    # 细化 5m
    fine = grid_points(best[:2], 5, half=50)
    best, _ = run_grid(fine, pts1, workers)
    print(f"    [细化 5m] {len(fine)} 点 → "
          f"S2=({best[0]:.1f}, {best[1]:.1f}) A={best[2]:.3f} m²")

    # 细化 1m
    fine2 = grid_points(best[:2], 1, half=5)
    best, _ = run_grid(fine2, pts1, workers)
    print(f"    [细化 1m] {len(fine2)} 点 → "
          f"S2=({best[0]:.1f}, {best[1]:.1f}) A={best[2]:.3f} m²")

    # 边界补查
    if best[1] <= 5.0:
        xs = np.arange(x_min + cell / 2, x_max, 1.0)
        edge = np.column_stack([xs, np.full_like(xs, y_min + cell / 2)])
        eb, _ = run_grid(edge, pts1, workers)
        if eb[2] < best[2]:
            best = eb
        print(f"    [边界补查] {len(edge)} 点 → "
              f"S2=({best[0]:.1f}, {best[1]:.1f}) A={best[2]:.3f} m²")

    # 邻域验证
    nb = [best[:2] + np.array([dx, dy])
          for dx in (-1, 0, 1) for dy in (-1, 0, 1)
          if not (dx == 0 and dy == 0)]
    _, nb_arr = run_grid(np.asarray(nb), pts1, workers)
    if nb_arr[:, 2].min() >= best[2] - eps:
        print("    [验证] 8 个 ±1m 邻域均更大，局部最小确认")

    elapsed = time.perf_counter() - t_start
    print("=" * 56)
    print("最终结果：")
    print(f"  S2 坐标  : x = {best[0]:.3f} m,  y = {best[1]:.3f} m")
    print(f"  累加面积 : {best[2]:.6f} m²")
    print(f"  总耗时   : {elapsed:.1f} 秒 ({elapsed / 60:.2f} 分钟)")
    print("=" * 56)

    np.savetxt("strategy2_best.csv", np.asarray([best]), delimiter=",",
               header="s2_x_m,s2_y_m,accumulated_area_m2", comments="",
               fmt="%.3f,%.3f,%.6f")
    draw_result(sector, best)
    print("已保存: strategy2_best.csv / strategy2_result.png")


if __name__ == "__main__":
    main()