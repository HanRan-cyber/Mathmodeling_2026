import os
import time
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")   # 后台模式：不弹窗，直接存图片
import matplotlib.pyplot as plt
# pathlib：用来定位输出目录
from pathlib import Path

# 中文字体设置，避免图上中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 全局参数 =================
sector_radius = 1500.0          # 扇形半径（米）
half_angle_deg = 1.0            # 扇形半张角（度），总张角 = 2 度
beam_half_deg = 1.0             # 探照灯光束半宽（度）
cell_size = 1.0                 # 网格边长（米）
receive_distance_min = 1000.0   # 有效接收距离下限（米），r ≤ 1000 权重为 1
weight_at_r_1500 = 1.0 / 501.0  # 1500 米处权重
second_x_min, second_x_max = 700.0, 1500.0    # 第二段区域 x 范围
second_y_min, second_y_max = 0.0, 1500.0      # 第二段区域 y 范围
theta_steps = 200               # 数值积分角向分段数
chunk_size = 4000               # 批量计算时单批最大点数
epsilon = 1e-9                  # 浮点零容差
use_parallel = True             # 是否启用多进程

pts1_cache = None               # 缓存：第一段点 P
weights1_cache = None           # 缓存：第一段点权重（归一化后总和 = 1）

# 输出目录：与本文件父目录同级
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
result_png = output_dir / "full_pipeline_weighted_result.png"
result_csv = output_dir / "full_pipeline_weighted_best.csv"


# =====================================================================
# 第一步：扇形几何（画轮廓 + 公式算面积）
# =====================================================================

# 把扇形描成多边形（圆弧用 n_arc 个点近似），方便画图
def sector_polygon(radius=sector_radius, half_deg=half_angle_deg, n_arc=400):
    theta = np.linspace(np.radians(-half_deg), np.radians(half_deg), n_arc)
    arc = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    # 原点 + 弧上点 + 原点，围成闭合扇形
    return np.vstack([np.zeros((1, 2)), arc, np.zeros((1, 2))])


# 用公式算扇形面积（作为标准答案，用于校验）
def sector_area(radius=sector_radius, half_deg=half_angle_deg):
    return (2 * half_deg) / 360.0 * np.pi * radius ** 2


# =====================================================================
# 第二步：扇形内按 1m×1m 网格取点 + 距离加权
# =====================================================================

# 在扇形内铺 1m×1m 网格，取每个格子中心（落在扇形内才保留）
def sample_sector_points(radius=sector_radius, half_deg=half_angle_deg,
                         cell=cell_size):
    x_centers = np.arange(cell / 2, radius, cell)
    half_span = radius * np.tan(np.radians(half_deg))
    y_centers = np.arange(-half_span, half_span, cell)
    X, Y = np.meshgrid(x_centers, y_centers)
    r = np.hypot(X, Y)
    ang = np.degrees(np.arctan2(Y, X))
    mask = (r <= radius) & (np.abs(ang) <= half_deg)
    return np.column_stack([X[mask], Y[mask]])


# 按距离 r 算权重：r ≤ 1000 权重 1；1000~1500 线性递减到 1/501
def weight_of(r):
    w = np.ones_like(r)
    mask = r > receive_distance_min
    w[mask] = 1.0 - (r[mask] - receive_distance_min) / \
        (sector_radius - receive_distance_min) * (1.0 - weight_at_r_1500)
    return w


# 取第一段点 P 及其归一化权重，带缓存
def load_pts1():
    global pts1_cache, weights1_cache
    if pts1_cache is None:
        pts1_cache = sample_sector_points()
        r_pts = np.hypot(pts1_cache[:, 0], pts1_cache[:, 1])
        w = weight_of(r_pts)
        weights1_cache = w / w.sum()   # 归一化，总和 = 1
    return pts1_cache


# =====================================================================
# 第三步：第二段区域 S2 候选点（全量约 120 万个）
# =====================================================================

# 在第二段区域（x: 700~1500, y: 0~1500）铺 1m 网格取中心点
def s2_candidates(x_min=second_x_min, x_max=second_x_max,
                  y_min=second_y_min, y_max=second_y_max,
                  cell=cell_size):
    xs = np.arange(x_min + cell / 2, x_max, cell)
    ys = np.arange(y_min + cell / 2, y_max, cell)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


# =====================================================================
# 第四步：算"探照灯光束 ∩ 原扇形"的面积（批量版）
# =====================================================================

# 核心数学：对每束光沿扇形角向切片积分，得到被照亮的面积
def _batch_wedge_area(s2, alphas, n_theta=theta_steps):
    a1 = np.radians(alphas - beam_half_deg)[:, None]   # 光束下边射线
    a2 = np.radians(alphas + beam_half_deg)[:, None]   # 光束上边射线
    c1, s1 = np.cos(a1), np.sin(a1)
    c2, s2c = np.cos(a2), np.sin(a2)

    # 扇形按角度切 n_theta 刀
    th = np.linspace(-half_angle_deg, half_angle_deg, n_theta) * np.pi / 180
    ct, st = np.cos(th), np.sin(th)

    # 每刀上光束约束可化简为 B*r + A = 0 的形式
    B1 = c1 * st - s1 * ct
    B2 = c2 * st - s2c * ct
    A1 = s1 * s2[0] - c1 * s2[1]
    A2 = s2c * s2[0] - c2 * s2[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = -A1 / B1
        r2 = -A2 / B2

    # 根据 B 的符号确定每刀半径的上下界
    lo = np.maximum(np.zeros_like(B1),
                    np.maximum(np.where(B1 > 0, r1, -np.inf),
                               np.where(B2 < 0, r2, -np.inf)))
    hi = np.where(B1 < 0, np.minimum(np.full_like(B1, sector_radius), r1),
                  sector_radius)
    hi = np.where(B2 > 0, np.minimum(hi, r2), hi)
    lo = np.clip(lo, 0.0, sector_radius)
    hi = np.clip(hi, 0.0, sector_radius)

    # 每刀月牙面积 = (hi² − lo²) / 2
    seg = np.maximum((hi * hi - lo * lo) / 2.0, 0.0)

    # 处理退化的临界情况
    bad = ((np.abs(B1) < 1e-12) & (A1 < -1e-9)) | \
          ((np.abs(B2) < 1e-12) & (A2 > 1e-9))
    seg = np.where(bad, 0.0, seg)

    # 沿角向梯形积分
    return np.trapezoid(seg, th)


# 分批调用，防止一次性占内存太多
def batch_wedge_area(s2, alphas):
    out = np.empty(alphas.shape[0])
    for i in range(0, len(alphas), chunk_size):
        out[i:i + chunk_size] = _batch_wedge_area(s2, alphas[i:i + chunk_size])
    return out


# 给一个 S2，把所有方向的光照面积按对应点权加起来 → 加权累加面积
def eval_s2(s2):
    pts1 = load_pts1()
    v = pts1 - s2
    d = np.hypot(v[:, 0], v[:, 1])
    keep = d > epsilon
    if not keep.any():
        return (float(s2[0]), float(s2[1]), 0.0)
    alphas = np.degrees(np.arctan2(v[keep, 1], v[keep, 0]))
    areas = batch_wedge_area(s2, alphas)
    score = float((weights1_cache[keep] * areas).sum())
    return (float(s2[0]), float(s2[1]), score)


# =====================================================================
# 第五步：多分辨率搜索 → 找加权累加面积最小的 S2
# =====================================================================

# 生成待测 S2 点：half=None 全区域撒点；否则以 center 为中心 ±half 内撒点
def grid_points(center, step, half=None):
    if half is None:
        xs = np.arange(second_x_min + cell_size / 2, second_x_max, step)
        ys = np.arange(second_y_min + cell_size / 2, second_y_max, step)
    else:
        x0 = np.round((center[0] - half - cell_size / 2) / step) * step \
            + cell_size / 2
        y0 = np.round((center[1] - half - cell_size / 2) / step) * step \
            + cell_size / 2
        xs = np.arange(x0, center[0] + half + cell_size / 2 + 1e-9, step)
        ys = np.arange(y0, center[1] + half + cell_size / 2 + 1e-9, step)
        xs = xs[(xs >= second_x_min + cell_size / 2) &
                (xs <= second_x_max - cell_size / 2)]
        ys = ys[(ys >= second_y_min + cell_size / 2) &
                (ys <= second_y_max - cell_size / 2)]
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


# 给一批点求加权累加面积，返回最优点和全部结果
def run_grid(points, workers):
    if use_parallel and workers > 1 and len(points) > 8:
        with Pool(workers) as pool:
            res = pool.map(eval_s2, points)
    else:
        res = [eval_s2(p) for p in points]
    arr = np.asarray(res)
    i = int(np.argmin(arr[:, 2]))
    return arr[i], arr


# 画出扇形、第二段区域、最优点，保存成 PNG
def draw_result(sector, best):
    fig, ax = plt.subplots(figsize=(8, 7))
    bx, by = sector[:, 0], sector[:, 1]
    ax.fill(bx, by, color="steelblue", alpha=0.3, label="原扇形区域")
    ax.plot(bx, by, color="steelblue", linewidth=1.2)
    ax.plot([0, sector_radius], [0, 0], "r--", linewidth=1.2)
    ax.plot([second_x_min, second_x_max, second_x_max, second_x_min, second_x_min],
            [second_y_min, second_y_min, second_y_max, second_y_max, second_y_min],
            "g--", linewidth=1.5,
            label=f"第二段区域 x∈({second_x_min:g},{second_x_max:g}), "
                  f"y∈({second_y_min:g},{second_y_max:g})")
    ax.plot(best[0], best[1], "o", ms=12, color="#d64541", zorder=5)
    ax.annotate(f"S2_min=({best[0]:.1f}, {best[1]:.1f})\nA'={best[2]:.1f} m²",
                xy=(best[0], best[1]), xytext=(best[0] - 260, best[1] + 220),
                fontsize=11, fontweight="bold", color="#d64541",
                arrowprops=dict(arrowstyle="->", color="#d64541"))
    ax.plot(0, 0, "ko", ms=7)
    ax.annotate("S1 (0,0)", xy=(0, 0), xytext=(60, -130), fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("加权累加面积最小的 S2 点位置（完整流程·加权修正版）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-250, 1750)
    ax.set_ylim(-300, 1700)
    plt.tight_layout()
    plt.savefig(result_png, dpi=150)
    plt.close(fig)


# 主流程：建扇形 → 取点加权 → 多分辨率搜索 → 输出结果
def main():
    t_start = time.perf_counter()

    # ---------- 第 1 步：扇形 ----------
    print("[第 1 步] 建立坐标系与扇形区域 ...")
    sector = sector_polygon()
    area = sector_area()
    print(f"          S1=(0,0)，x 轴为中心线，±{half_angle_deg:g}°，"
          f"R={sector_radius:g} m，扇形面积={area:,.1f} m²")

    # ---------- 第 2 步：取点 + 加权 ----------
    print("[第 2 步] 扇形内 1m×1m 网格取点，按距离加权 ...")
    global pts1_cache, weights1_cache
    pts1_cache = sample_sector_points()
    r_pts = np.hypot(pts1_cache[:, 0], pts1_cache[:, 1])
    w_raw = weight_of(r_pts)
    weights1_cache = w_raw / w_raw.sum()
    n_in = int((r_pts <= receive_distance_min).sum())
    print(f"          第一段点 P 数量: {len(pts1_cache):,}"
          f"（密度 {len(pts1_cache) / area:.4f} 点/m²）")
    print(f"          [加权] r≤{receive_distance_min:g} m: {n_in:,} 点（权重=1）；"
          f"r>{receive_distance_min:g} m: {len(pts1_cache) - n_in:,} 点"
          f"（递减至 {weight_at_r_1500:.6f}）")

    # ---------- 第 3 步：第二段区域候选 ----------
    print("[第 3 步] 第二段区域候选 S2 ...")
    s2_all = s2_candidates()
    print(f"          全量候选 S2: {len(s2_all):,} 个"
          f"（区域面积 "
          f"{(second_x_max - second_x_min) * (second_y_max - second_y_min):,.0f} m²）")

    # ---------- 第 4/5 步：多分辨率搜索 ----------
    print("[第 4/5 步] 多分辨率搜索加权累加面积最小的 S2 ...")
    n_workers = max(1, os.cpu_count() or 1)
    print(f"          并行核数: {n_workers}")

    coarse = grid_points(None, 50)
    best, _ = run_grid(coarse, n_workers)
    print(f"          [粗筛 50m] {len(coarse):,} 点 → "
          f"S2=({best[0]:.1f}, {best[1]:.1f}) A'={best[2]:,.3f} m²")

    fine = grid_points(best[:2], 5, half=50)
    best, _ = run_grid(fine, n_workers)
    print(f"          [细化 5m ] {len(fine):,} 点 → "
          f"S2=({best[0]:.1f}, {best[1]:.1f}) A'={best[2]:,.3f} m²")

    fine2 = grid_points(best[:2], 1, half=5)
    best, _ = run_grid(fine2, n_workers)
    print(f"          [细化 1m ] {len(fine2):,} 点 → "
          f"S2=({best[0]:.1f}, {best[1]:.1f}) A'={best[2]:,.3f} m²")

    # 最优点贴近 y=0 边界时，沿边界补查一次
    if best[1] <= 5.0:
        xs = np.arange(second_x_min + cell_size / 2, second_x_max, 1.0)
        edge = np.column_stack([xs, np.full_like(xs, second_y_min + cell_size / 2)])
        eb, _ = run_grid(edge, n_workers)
        if eb[2] < best[2]:
            best = eb
        print(f"          [边界补查] {len(edge):,} 点 → "
              f"S2=({best[0]:.1f}, {best[1]:.1f}) A'={best[2]:,.3f} m²")

    # 验证：最优点 8 邻域是否都更大
    nb = [best[:2] + np.array([dx, dy])
          for dx in (-1, 0, 1) for dy in (-1, 0, 1)
          if not (dx == 0 and dy == 0)]
    _, nb_arr = run_grid(np.asarray(nb), n_workers)
    if nb_arr[:, 2].min() >= best[2] - epsilon:
        print("          [验证] 最优 S2 的 8 个 ±1m 邻域均更大 → 局部最小确认")

    # ---------- 输出 ----------
    elapsed = time.perf_counter() - t_start
    print("=" * 56)
    print("最终结果：加权累加面积（期望覆盖面积）最小的 S2")
    print(f"  S2 坐标       : x = {best[0]:.3f} m,  y = {best[1]:.3f} m")
    print(f"  加权累加面积  : {best[2]:,.6f} m²  （权重 wᵢ=w(rᵢ)，Σwᵢ=1）")
    print(f"  总耗时        : {elapsed:.1f} 秒（{elapsed / 60:.2f} 分钟）")
    print("=" * 56)

    np.savetxt(result_csv, np.asarray([best]), delimiter=",",
               header="s2_x_m,s2_y_m,weighted_area_m2", comments="",
               fmt="%.3f,%.3f,%.6f")
    draw_result(sector, best)
    print(f"已保存: {result_csv}")
    print(f"已保存: {result_png}")


if __name__ == "__main__":
    main()