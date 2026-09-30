import heapq
import os
import sys
import time

# Route to task2 for core logic
source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(source_dir, "requirement_2"))

from algorithms import SokobanState, get_successors, load_map_file, parse_map
from heuristic import bipartite_matching_heuristic, precompute_maze_distances


def a_star_search(initial_state, targets, walls, dist_table):
    # Solve Sokoban using A*
    initial_h = bipartite_matching_heuristic(initial_state, targets, walls, dist_table)
    frontier = []
    heapq.heappush(frontier, (initial_h, 0, 0, initial_state, []))

    explored = set()
    counter = 1
    nodes_expanded = 0

    while frontier:
        f_cost, cost, _, current_state, path = heapq.heappop(frontier)

        if current_state.B == targets:
            return path, cost, nodes_expanded

        if current_state not in explored:
            explored.add(current_state)
            nodes_expanded += 1

            for successor, action, step_cost in get_successors(current_state, walls):
                if successor not in explored:
                    h_cost = bipartite_matching_heuristic(
                        successor, targets, walls, dist_table
                    )
                    if h_cost == float("inf"):
                        continue

                    new_cost = cost + step_cost
                    f_new = new_cost + h_cost
                    heapq.heappush(
                        frontier, (f_new, new_cost, counter, successor, path + [action])
                    )
                    counter += 1

    return None, 0, nodes_expanded


if __name__ == "__main__":
    # Test A* logic
    layout_content = load_map_file("example_map.txt")
    p0, b0, targets, walls, width, height = parse_map(layout_content)
    init_state = SokobanState(p0, b0)

    print("Precomputing maze distances from targets...")
    dist_table = precompute_maze_distances(targets, walls, width, height)

    print("\n--- Running A* Search ---")
    start_time = time.time()
    a_star_path, a_star_cost, a_star_expanded = a_star_search(
        init_state, targets, walls, dist_table
    )
    a_star_duration = time.time() - start_time

    if a_star_path:
        print(f"Status: Solved in {a_star_duration:.4f}s")
        print(f"Total Cost: {a_star_cost}")
        print(f"Nodes Expanded: {a_star_expanded}")
        print(f"Actions ({len(a_star_path)} steps): {a_star_path}")
    else:
        print("Status: No solution found.")
