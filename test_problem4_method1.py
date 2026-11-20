#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题4 方法1 离线模拟验证（默认跑100次，统计成功率）。

运行：
    python test_problem4_method1.py
    python test_problem4_method1.py --cases 200 --directional-ratio 1.0
输出：
    output/method1_results.json
"""

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace

OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "method1_results.json"


def load_module():
    names = ["problem4_method1 2.py", "problem4_method1.py"]
    roots = [
        pathlib.Path("src"),
        pathlib.Path("b3_logs"),
        pathlib.Path(__file__).parent / "src",
        pathlib.Path(__file__).parent,
        pathlib.Path("."),
    ]
    for root in roots:
        for n in names:
            p = root / n
            if p.exists():
                print("加载主程序：%s" % p.resolve())
                spec = importlib.util.spec_from_file_location("p4m1", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    print("找不到方法1主程序，尝试过：")
    for root in roots:
        for n in names:
            print("  -", (root / n).resolve())
    sys.exit(1)


P4 = load_module()
Client = P4.Client
Solver = P4.Solver
distance = P4.distance


@dataclass
class Source:
    channel: int
    position: tuple
    radius: float
    is_directional: bool = False
    orientation: float = 0.0
    cleared: bool = False


def random_case(seed, count=None, directional_ratio=0.5,
                boundary=False, min_range=False, clustered=False):
    rng = random.Random(seed)
    count = rng.randint(10, 16) if count is None else count
    sources = []
    for ch in rng.sample(range(1, 21), count):
        a = rng.uniform(0, 2 * math.pi)
        if clustered:
            cx, cy = 1400, 200
            p = (cx + rng.uniform(-50, 50), cy + rng.uniform(-50, 50))
        else:
            r = 1800 if boundary else 1800 * math.sqrt(rng.random())
            p = (r * math.cos(a), r * math.sin(a))
        is_dir = rng.random() < directional_ratio
        orientation = rng.uniform(0, 360) if is_dir else 0.0
        sources.append(Source(
            channel=ch,
            position=p,
            radius=1000 if min_range else rng.uniform(1000, 1500),
            is_directional=is_dir,
            orientation=orientation,
        ))
    return sources


class MockSimulator:
    def __init__(self, sources, seed, noise="hash", duration=1200.0):
        self.sources = {s.channel: replace(s) for s in sources}
        self.seed = seed
        self.noise = noise
        self.duration = duration
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_us = 0
        self.entered = False
        self.closed = False
        self.cache = {}
        self.actions = []

    def error_at(self, position, channel):
        key = ("%d|%d|%.12g|%.12g" % (self.seed, channel, *position)).encode()
        z = int.from_bytes(
            hashlib.blake2b(key, digest_size=8).digest(), "big"
        ) / (2 ** 64 - 1)
        if self.noise == "zero":
            return 0.0
        if self.noise == "positive":
            return 1.0
        if self.noise == "negative":
            return -1.0
        if self.noise == "extreme":
            return 1.0 if z > 0.5 else -1.0
        return 2 * z - 1

    def in_coverage_angle(self, source, position):
        if not source.is_directional:
            return True
        dx = position[0] - source.position[0]
        dy = position[1] - source.position[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return True
        angle_to_pos = math.degrees(math.atan2(dy, dx)) % 360
        diff = abs(angle_to_pos - source.orientation) % 360
        diff = min(diff, 360 - diff)
        return diff <= 90.0 + 1e-6

    def reply(self, accepted, **fields):
        return dict(accepted=accepted, real_timestamp_ms=0,
                    virtual_time_s=self.virtual_us / 1e6 if accepted else 0,
                    **fields)

    def request(self, path, payload, deadline=None):
        key = payload["request_id"]
        signature = (path, json.dumps(payload, sort_keys=True, allow_nan=False))
        if key in self.cache:
            old_sig, response = self.cache[key]
            if old_sig != signature:
                raise RuntimeError("409: idempotency mismatch")
            return dict(response)

        if payload["arena_id"] != "default":
            return self.reply(False)
        if self.closed:
            return self.reply(False)

        if path == "/enter":
            if self.entered:
                return self.reply(False)
            self.entered = True
            response = self.reply(True,
                                  max_virtual_duration_s=360000,
                                  max_real_duration_s=1200,
                                  remaining_real_duration_s=self.duration)
        elif not self.entered:
            return self.reply(False)
        elif path == "/exit":
            self.closed = True
            response = self.reply(True, exit_reason="user_exit")
        else:
            position = (payload["position"]["x"], payload["position"]["y"])
            ch = payload["channel"]

            duration = math.hypot(position[0] - self.position[0],
                                  position[1] - self.position[1]) / 5
            self.position = position

            source = self.sources.get(ch)
            live = source is not None and not source.cleared
            d = distance(position, source.position) if live else math.inf
            in_cov = self.in_coverage_angle(source, position) if live else False

            if path == "/measure":
                duration += 5 + int(ch != self.channel)
                self.channel = ch
                if not live or d > source.radius or not in_cov:
                    fields = dict(measure_result="no_signal")
                elif d <= 5:
                    fields = dict(measure_result="near")
                else:
                    angle = math.degrees(math.atan2(
                        source.position[1] - position[1],
                        source.position[0] - position[0]))
                    angle = round((angle + self.error_at(position, ch)) % 360, 2) % 360
                    fields = dict(measure_result="direction", svd_deg=angle)
            elif path == "/clear":
                success = live and d <= 20
                duration += 5 if success else 3
                if success:
                    source.cleared = True
                fields = dict(clear_result="success" if success
                              else "no_target_in_range")
            else:
                raise RuntimeError("unknown path: %s" % path)

            self.virtual_us += round(duration * 1e6)
            response = self.reply(True, **fields)

        self.cache[key] = (signature, dict(response))
        self.actions.append((path, payload, dict(response)))
        return response


def run_case(sources, seed, noise="hash", verbose=False):
    simulator = MockSimulator(sources, seed, noise)
    client = Client(simulator, "", log_path=None)  #空字符串，运行时通过--robot-id传入
    solver = Solver(client, optimized=True, verbose=verbose)

    start = time.perf_counter()
    error = None
    try:
        resp = client.enter()
        if not resp.get("accepted"):
            raise RuntimeError("enter rejected: %s" % resp)
        solver.solve()
        client.exit()
    except Exception as ex:
        error = "%s: %s" % (type(ex).__name__, ex)

    result = solver.summary()
    actual_cleared = sum(1 for s in simulator.sources.values() if s.cleared)
    result.update(
        seed=seed,
        true_source_count=len(sources),
        directional_count=sum(1 for s in sources if s.is_directional),
        actual_cleared=actual_cleared,
        runtime_s=time.perf_counter() - start,
        error=error,
    )
    result["clearance_ratio"] = (actual_cleared / len(sources)
                                 if sources else 1.0)
    result["independently_verified_complete"] = (
        bool(result.get("complete", False))
        and actual_cleared == len(sources)
        and not error)
    return result


# ========== 汇总统计 ==========
def summarize(results, cases):
    n = len(results)
    ok = sum(1 for r in results if r["independently_verified_complete"])
    total_src = sum(r["true_source_count"] for r in results)
    total_cleared = sum(r["actual_cleared"] for r in results)
    ratios = [r["clearance_ratio"] for r in results]
    vtimes = [r.get("total_virtual_s", 0) for r in results]
    runtimes = [r["runtime_s"] for r in results]
    measures = [r.get("measurements", 0) for r in results]
    dists = [r.get("distance_m", 0) for r in results]  #结果里这个字段叫 distance_m，之前取错了名字，汇总里的平均距离一直是 0

    summary = {
        "cases": n,
        "completed_cases": ok,
        "success_rate": ok / n if n else 0.0,
        "total_sources": total_src,
        "total_cleared": total_cleared,
        "overall_clearance_ratio": (total_cleared / total_src
                                    if total_src else 0.0),
        "avg_clearance_ratio": statistics.mean(ratios) if ratios else 0.0,
        "min_clearance_ratio": min(ratios) if ratios else 0.0,
        "avg_virtual_time_s": statistics.mean(vtimes) if vtimes else 0.0,
        "median_virtual_time_s": statistics.median(vtimes) if vtimes else 0.0,
        "avg_runtime_s": statistics.mean(runtimes) if runtimes else 0.0,
        "avg_measurements": statistics.mean(measures) if measures else 0.0,
        "avg_distance_m": statistics.mean(dists) if dists else 0.0,
        "error_cases": sum(1 for r in results if r["error"]),
    }
    return summary


def print_summary(summary):
    print("=" * 64)
    print("汇总")
    print("=" * 64)
    print("总案例数        : %d" % summary["cases"])
    print("完全成功案例    : %d" % summary["completed_cases"])
    print("成功率          : %.2f%%" % (summary["success_rate"] * 100))
    print("清除目标总数    : %d/%d" % (summary["total_cleared"],
                                       summary["total_sources"]))
    print("整体清除率      : %.2f%%" % (summary["overall_clearance_ratio"] * 100))
    print("平均清除率      : %.2f%%" % (summary["avg_clearance_ratio"] * 100))
    print("最低清除率      : %.2f%%" % (summary["min_clearance_ratio"] * 100))
    print("平均虚拟时间    : %.1f s" % summary["avg_virtual_time_s"])
    print("中位数虚拟时间  : %.1f s" % summary["median_virtual_time_s"])
    print("平均测量次数    : %.1f" % summary["avg_measurements"])
    print("平均移动距离    : %.0f m" % summary["avg_distance_m"])
    print("平均运行时间    : %.2f s" % summary["avg_runtime_s"])
    print("异常案例数      : %d" % summary["error_cases"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100,
                        help="案例数（默认100）")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--directional-ratio", type=float, default=0.5)
    parser.add_argument("--noise", default="hash")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("问题4 方法1 离线模拟验证")
    print("=" * 64)
    print("案例数      : %d" % args.cases)
    print("起始种子    : %d" % args.seed)
    print("定向源比例  : %.0f%%" % (args.directional_ratio * 100))
    print("噪声模型    : %s" % args.noise)
    print("输出文件    : %s" % OUTPUT_PATH.resolve())
    print("=" * 64)

    results = []
    t0 = time.perf_counter()
    for i in range(args.cases):
        seed = args.seed + i
        sources = random_case(seed,
                              directional_ratio=args.directional_ratio)
        r = run_case(sources, seed, args.noise, args.verbose)
        results.append(r)

        mark = "✓" if r["independently_verified_complete"] else "✗"
        print("[%3d/%3d] seed=%-10d src=%-2d dir=%-2d 清 %2d/%2d %s 虚拟 %6.0fs 用时 %5.2fs"
              % (i + 1, args.cases, seed,
                 r["true_source_count"], r["directional_count"],
                 r["actual_cleared"], r["true_source_count"], mark,
                 r.get("total_virtual_s", 0), r["runtime_s"]),
              flush=True)
        if not r["independently_verified_complete"] and r["error"]:
            print("        error: %s" % r["error"])

        # 每 10 个案例打印一次阶段成功率
        if (i + 1) % 10 == 0:
            done = results[-10:]
            ok10 = sum(1 for x in done if x["independently_verified_complete"])
            print("  --- 最近10个案例成功率 %d/10 = %.0f%% | 累计 %.2fs ---"
                  % (ok10, ok10 * 10.0,
                     time.perf_counter() - t0),
                  flush=True)

    summary = summarize(results, args.cases)
    print()
    print_summary(summary)

    OUTPUT_PATH.write_text(json.dumps({
        "method": "method1",
        "params": vars(args),
        "summary": summary,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n结果已保存到 %s" % OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()