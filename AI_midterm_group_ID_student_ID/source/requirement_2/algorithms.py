import heapq
import os
import time


class SokobanState:
    def __init__(self, P, B):
        self.P = P
        self.B = frozenset(B)

    def __eq__(self, other):
        return self.P == other.P and self.B == other.B

    def __hash__(self):
        return hash((self.P, self.B))


def parse_map(layout_str):
    # Parse map into components
    lines = layout_str.strip("\r\n").split("\n")
    height = len(lines)
    width = max(len(line) for line in lines)

    agent_pos = None
    boxes = set()
    targets = set()
    walls = set()

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            coord = (x, y)
            if char == "%":
                walls.add(coord)
            elif char == "A":
                agent_pos = coord
            elif char == "B":
                boxes.add(coord)
            elif char == "D":
                targets.add(coord)
            elif char == "C":
                boxes.add(coord)
                targets.add(coord)

    return agent_pos, frozenset(boxes), frozenset(targets), walls, width, height


def load_map_file(filename="example_map.txt"):
    # Read map file
    base_dir = os.path.dirname(__file__)
    map_path = os.path.join(base_dir, filename)
    with open(map_path, "r", encoding="utf-8") as file:
        return file.read()


def get_successors(state, walls):
    # Apply actions and box-pushing logic
    actions = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0)}

    for action_name, (dx, dy) in actions.items():
        new_px, new_py = state.P[0] + dx, state.P[1] + dy
        new_p = (new_px, new_py)

        if new_p not in walls and new_p not in state.B:
            yield (SokobanState(new_p, state.B), action_name, 1)

        elif new_p in state.B:
            push_px, push_py = new_px + dx, new_py + dy
            push_p = (push_px, push_py)

            if push_p not in walls and push_p not in state.B:
                new_B = set(state.B)
                new_B.remove(new_p)
                new_B.add(push_p)
                yield (SokobanState(new_p, new_B), action_name, 1)


def uniform_cost_search(initial_state, targets, walls):
    # Solve Sokoban using UCS
    frontier = []
    heapq.heappush(frontier, (0, 0, initial_state, []))

    explored = set()
    counter = 1
    nodes_expanded = 0

    while frontier:
        cost, _, current_state, path = heapq.heappop(frontier)

        if current_state.B == targets:
            return path, cost, nodes_expanded

        if current_state not in explored:
            explored.add(current_state)
            nodes_expanded += 1

            for successor, action, step_cost in get_successors(current_state, walls):
                if successor not in explored:
                    new_cost = cost + step_cost
                    heapq.heappush(
                        frontier, (new_cost, counter, successor, path + [action])
                    )
                    counter += 1

    return None, 0, nodes_expanded


if __name__ == "__main__":
    # Test UCS logic
    layout_content = load_map_file("example_map.txt")
    p0, b0, targets, walls, width, height = parse_map(layout_content)
    init_state = SokobanState(p0, b0)

    print("--- Running Uniform Cost Search ---")
    start_time = time.time()
    ucs_path, ucs_cost, ucs_expanded = uniform_cost_search(init_state, targets, walls)
    ucs_duration = time.time() - start_time

    if ucs_path:
        print(f"Status: Solved in {ucs_duration:.4f}s")
        print(f"Total Cost: {ucs_cost}")
        print(f"Nodes Expanded: {ucs_expanded}")
    else:
        print("Status: No solution found.")
