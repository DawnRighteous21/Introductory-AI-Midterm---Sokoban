import argparse
import csv
import heapq
import multiprocessing as mp
import os
import statistics
import sys
import time
import tracemalloc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(BASE_DIR)

req2_dir = os.path.join(SOURCE_DIR, "requirement_2")
req4_dir = os.path.join(SOURCE_DIR, "requirement_4")

sys.path.append(req2_dir)
sys.path.append(req4_dir)

import algorithms
import heuristic
from algorithms import SokobanState, parse_map
from heuristic import precompute_maze_distances

MAP_DIR = os.path.join(BASE_DIR, "maps")
RESULT_DIR = os.path.join(BASE_DIR, "results")

ALGORITHMS = ["UCS", "A*"]


def collect_maps(selected_names):
    maps = []

    example_path = os.path.join(BASE_DIR, "example_map.txt")
    if os.path.exists(example_path):
        maps.append(("example_map", example_path))

    if os.path.isdir(MAP_DIR):
        for file_name in sorted(os.listdir(MAP_DIR)):
            if file_name.endswith(".txt"):
                map_name = os.path.splitext(file_name)[0]
                maps.append((map_name, os.path.join(MAP_DIR, file_name)))

    if selected_names:
        maps = [item for item in maps if item[0] in selected_names]

    return maps


def load_problem(map_path):
    with open(map_path, "r", encoding="utf-8") as file:
        layout = file.read()

    agent, boxes, targets, walls, width, height = parse_map(layout)
    initial_state = SokobanState(agent, boxes)
    floor_count = 0
    for y in range(height):
        for x in range(width):
            if (x, y) not in walls:
                floor_count += 1

    info = {
        "width": width,
        "height": height,
        "boxes": len(boxes),
        "floor": floor_count,
    }
    return initial_state, targets, walls, width, height, info


class CountingHeapq:
    def __init__(self):
        self.pushes = 0
        self.max_frontier = 0

    def heappush(self, heap, item):
        heapq.heappush(heap, item)
        self.pushes += 1
        if len(heap) > self.max_frontier:
            self.max_frontier = len(heap)

    def heappop(self, heap):
        return heapq.heappop(heap)


class CountingHeuristic:
    def __init__(self, function):
        self.function = function
        self.calls = 0
        self.total_time = 0.0
        self.pruned = 0

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        value = self.function(*args, **kwargs)
        self.total_time += time.perf_counter() - start
        self.calls += 1
        if value == float("inf"):
            self.pruned += 1
        return value


def run_search(algorithm, initial_state, targets, walls, dist_table):
    if algorithm == "UCS":
        return algorithms.uniform_cost_search(initial_state, targets, walls)
    return algorithms.a_star_search(initial_state, targets, walls, dist_table)


def timing_worker(map_path, algorithm, queue):
    initial_state, targets, walls, width, height, _ = load_problem(map_path)

    pre_start = time.perf_counter()
    dist_table = None
    if algorithm == "A*":
        dist_table = precompute_maze_distances(targets, walls, width, height)
    pre_time = time.perf_counter() - pre_start

    start = time.perf_counter()
    path, cost, expanded = run_search(
        algorithm, initial_state, targets, walls, dist_table
    )
    search_time = time.perf_counter() - start

    queue.put(
        {
            "solved": path is not None,
            "cost": cost,
            "expanded": expanded,
            "search_time": search_time,
            "pre_time": pre_time,
        }
    )


def profile_worker(map_path, algorithm, queue):
    initial_state, targets, walls, width, height, _ = load_problem(map_path)

    counting_heapq = CountingHeapq()
    counting_h = CountingHeuristic(heuristic.bipartite_matching_heuristic)
    algorithms.heapq = counting_heapq
    algorithms.bipartite_matching_heuristic = counting_h

    generated = {"count": 0}
    original_successors = algorithms.get_successors

    def counting_successors(state, walls_set):
        for item in original_successors(state, walls_set):
            generated["count"] += 1
            yield item

    algorithms.get_successors = counting_successors

    tracemalloc.start()
    dist_table = None
    if algorithm == "A*":
        dist_table = precompute_maze_distances(targets, walls, width, height)
    search_start = time.perf_counter()
    path, cost, expanded = run_search(
        algorithm, initial_state, targets, walls, dist_table
    )
    search_time = time.perf_counter() - search_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queue.put(
        {
            "generated": generated["count"],
            "pushed": counting_heapq.pushes + 1,
            "max_frontier": counting_heapq.max_frontier,
            "peak_mem_mb": peak_bytes / (1024 * 1024),
            "h_calls": counting_h.calls,
            "h_time": counting_h.total_time,
            "search_time": search_time,
            "h_pruned": counting_h.pruned,
            "path_len": len(path) if path is not None else 0,
        }
    )


def run_in_child(worker, map_path, algorithm, timeout):
    queue = mp.Queue()
    process = mp.Process(target=worker, args=(map_path, algorithm, queue))
    wall_start = time.perf_counter()
    process.start()

    result = None
    try:
        result = queue.get(timeout=timeout)
    except Exception:
        result = None

    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()

    if result is not None:
        result["wall"] = time.perf_counter() - wall_start
    return result


def effective_branching_factor(nodes, depth):
    if depth <= 0 or nodes <= 0:
        return 0.0

    def total(b):
        if abs(b - 1.0) < 1e-12:
            return depth + 1
        return (b ** (depth + 1) - 1) / (b - 1)

    target = nodes + 1
    low = 1.0
    high = 10.0
    for _ in range(200):
        mid = (low + high) / 2
        if total(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def benchmark(maps, repeats, timeout):
    rows = []
    for map_name, map_path in maps:
        _, _, _, _, _, info = load_problem(map_path)
        print(
            f"\n=== {map_name}  ({info['width']}x{info['height']}, "
            f"{info['boxes']} boxes, {info['floor']} floor cells) ==="
        )

        for algorithm in ALGORITHMS:
            row = {"map": map_name, "algorithm": algorithm}
            row.update(info)

            times = []
            timing = None
            for _ in range(repeats):
                timing = run_in_child(timing_worker, map_path, algorithm, timeout)
                if timing is None:
                    break
                times.append(timing["search_time"])

            if timing is None:
                row["status"] = "TIMEOUT"
                print(f"  {algorithm:<4}: TIMEOUT (> {timeout}s)")
                rows.append(row)
                continue

            profile = run_in_child(profile_worker, map_path, algorithm, timeout * 4)

            row["status"] = "SOLVED" if timing["solved"] else "NO SOLUTION"
            row["cost"] = timing["cost"]
            row["expanded"] = timing["expanded"]
            row["time_median_s"] = statistics.median(times)
            row["time_min_s"] = min(times)
            row["time_max_s"] = max(times)
            row["precompute_s"] = timing["pre_time"]
            row["us_per_expansion"] = (
                1e6 * row["time_median_s"] / max(timing["expanded"], 1)
            )
            row["ebf"] = effective_branching_factor(timing["expanded"], timing["cost"])
            row["max_explored"] = timing["expanded"]

            if profile is not None:
                row["generated"] = profile["generated"]
                row["pushed"] = profile["pushed"]
                row["max_frontier"] = profile["max_frontier"]
                row["peak_mem_mb"] = profile["peak_mem_mb"]
                row["h_calls"] = profile["h_calls"]
                row["h_pruned"] = profile["h_pruned"]
                if algorithm == "A*" and profile["h_calls"] > 0:
                    row["h_time_share_pct"] = (
                        100.0 * profile["h_time"] / max(profile["search_time"], 1e-9)
                    )

            print(
                f"  {algorithm:<4}: cost={row['cost']:<4} expanded={row['expanded']:<9} "
                f"time={row['time_median_s']:.4f}s  "
                f"frontier_max={row.get('max_frontier', '-')}  "
                f"peak_mem={row.get('peak_mem_mb', 0):.2f} MB"
            )
            rows.append(row)
    return rows


CSV_COLUMNS = [
    "map",
    "algorithm",
    "status",
    "width",
    "height",
    "boxes",
    "floor",
    "cost",
    "expanded",
    "generated",
    "pushed",
    "max_frontier",
    "max_explored",
    "time_median_s",
    "time_min_s",
    "time_max_s",
    "precompute_s",
    "us_per_expansion",
    "ebf",
    "peak_mem_mb",
    "h_calls",
    "h_pruned",
    "h_time_share_pct",
]


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in CSV_COLUMNS:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = round(value, 6)
                clean[key] = value
            writer.writerow(clean)


def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row = {}
            for key, value in raw.items():
                if value == "":
                    continue
                try:
                    number = float(value)
                    if number.is_integer() and "." not in value and "e" not in value:
                        row[key] = int(number)
                    else:
                        row[key] = number
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def make_charts(rows, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> charts skipped (pip install matplotlib)")
        return

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.edgecolor": "black",
            "axes.grid": True,
            "grid.color": "#bbbbbb",
            "grid.linestyle": ":",
        }
    )

    map_names = []
    for row in rows:
        if row["map"] not in map_names:
            map_names.append(row["map"])

    def metric(algorithm, key):
        values = []
        for name in map_names:
            value = None
            for row in rows:
                if row["map"] == name and row["algorithm"] == algorithm:
                    value = row.get(key)
            values.append(value if isinstance(value, (int, float)) else 0)
        return values

    styles = {
        "UCS": {"color": "#d9d9d9", "edgecolor": "black", "hatch": "///"},
        "A*": {"color": "#404040", "edgecolor": "black", "hatch": ""},
    }
    short_labels = [
        name.split("_", 1)[-1].replace("_", "\n") if name.startswith("map") else name
        for name in map_names
    ]

    def grouped_bar(key, title, ylabel, file_name, log_scale):
        fig, ax = plt.subplots(figsize=(8, 6))
        width = 0.38
        positions = list(range(len(map_names)))
        for index, algorithm in enumerate(ALGORITHMS):
            offset = (index - 0.5) * width
            xs = [p + offset for p in positions]
            values = metric(algorithm, key)
            bars = ax.bar(xs, values, width, label=algorithm, **styles[algorithm])
            for bar, value in zip(bars, values):
                if value > 0:
                    if value >= 1e6:
                        text = f"{value / 1e6:.1f}M"
                    elif value >= 1e4:
                        text = f"{value / 1e3:.0f}k"
                    elif value >= 1000:
                        text = f"{value:,.0f}"
                    elif value >= 10:
                        text = f"{value:.0f}"
                    elif value >= 1e-4:
                        text = f"{value:.2g}"
                    else:
                        text = f"{value:.6f}".rstrip("0")
                    ax.annotate(
                        text,
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        rotation=90,
                        xytext=(0, 3),
                        textcoords="offset points",
                    )
        ax.set_xticks(positions)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        low, high = ax.get_ylim()
        if log_scale:
            ax.set_ylim(low, high * 12)
        else:
            ax.set_ylim(low, high * 1.35)
        ax.legend(frameon=True, edgecolor="black", loc="upper left")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, file_name), dpi=200)
        plt.close(fig)

    grouped_bar(
        "expanded",
        "Nodes expanded (log scale)",
        "Nodes expanded",
        "chart_nodes_expanded.png",
        True,
    )
    grouped_bar(
        "time_median_s",
        "Search time (log scale)",
        "Seconds (median)",
        "chart_runtime.png",
        True,
    )
    grouped_bar(
        "peak_mem_mb",
        "Peak memory (log scale)",
        "MB (tracemalloc peak)",
        "chart_peak_memory.png",
        True,
    )
    grouped_bar(
        "max_frontier",
        "Maximum frontier size (log scale)",
        "Nodes in frontier",
        "chart_max_frontier.png",
        True,
    )
    grouped_bar(
        "us_per_expansion",
        "Time per expanded node",
        "Microseconds / node",
        "chart_time_per_node.png",
        False,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ucs_nodes = metric("UCS", "expanded")
    a_nodes = metric("A*", "expanded")
    ucs_time = metric("UCS", "time_median_s")
    a_time = metric("A*", "time_median_s")
    node_ratio = []
    time_ratio = []
    for un, an, ut, at in zip(ucs_nodes, a_nodes, ucs_time, a_time):
        node_ratio.append(un / an if an else 0)
        time_ratio.append(ut / at if at else 0)
    positions = list(range(len(map_names)))
    width = 0.38
    ax.bar(
        [p - width / 2 for p in positions],
        node_ratio,
        width,
        label="Nodes ratio (UCS / A*)",
        color="#bfbfbf",
        edgecolor="black",
        hatch="..",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        time_ratio,
        width,
        label="Time ratio (UCS / A*)",
        color="#404040",
        edgecolor="black",
    )
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Break-even (ratio = 1)",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_title("UCS / A* ratio (> 1 means A* is better)")
    ax.set_ylabel("Ratio")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_ratio.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    header = ["Map", "Alg.", "Cost", "Expanded", "Time (ms)", "Mem (MB)", "EBF"]
    cells = []
    for row in rows:
        if row.get("status") != "SOLVED":
            cells.append(
                [
                    row["map"],
                    row["algorithm"],
                    "-",
                    "-",
                    row.get("status", "-"),
                    "-",
                    "-",
                ]
            )
            continue
        cells.append(
            [
                row["map"].replace("map", "").split("_")[0] or row["map"],
                row["algorithm"],
                str(row["cost"]),
                f"{row['expanded']:,}",
                f"{row['time_median_s'] * 1000:.2f}",
                f"{row.get('peak_mem_mb', 0):.2f}",
                f"{row['ebf']:.3f}",
            ]
        )
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", weight="bold")
        elif cells[r - 1][1] == "A*":
            cell.set_facecolor("#e6e6e6")
    ax.set_title("UCS vs A* - summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_table.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Req 3: UCS vs A* benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats per run")
    parser.add_argument("--timeout", type=float, default=180, help="seconds per run")
    parser.add_argument("--maps", nargs="*", default=None, help="map names to run")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="keep rows of other maps from the existing results.csv",
    )
    args = parser.parse_args()

    maps = collect_maps(args.maps)
    if not maps:
        print("No maps found. Put .txt maps in the maps/ folder.")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = benchmark(maps, args.repeats, args.timeout)

    csv_path = os.path.join(RESULT_DIR, "results.csv")
    if args.merge:
        new_maps = set()
        for row in rows:
            new_maps.add(row["map"])
        old_rows = []
        for row in read_csv(csv_path):
            if row["map"] not in new_maps:
                old_rows.append(row)
        rows = old_rows + rows
        for row in rows:
            if "max_explored" not in row and "expanded" in row:
                row["max_explored"] = row["expanded"]
        rows.sort(key=lambda r: (r["map"], ALGORITHMS.index(r["algorithm"])))
    write_csv(rows, csv_path)
    make_charts(rows, RESULT_DIR)
    print(f"\nSaved: {csv_path}")
    print(f"Charts: {RESULT_DIR}")


if __name__ == "__main__":
    main()
