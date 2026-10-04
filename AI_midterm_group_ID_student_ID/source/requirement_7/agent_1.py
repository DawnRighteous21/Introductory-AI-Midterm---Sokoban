import heapq
import itertools
import time
from collections import deque

# Global caches for static BFS distances to avoid recalculating per state
GLOBAL_TARGET_DISTANCES = {}
GLOBAL_WALLS_HASH = None


class LocalState:
    def __init__(self, p, boxes):
        self.p = p
        self.boxes = frozenset(boxes)

    def __eq__(self, other):
        return self.p == other.p and self.boxes == other.boxes

    def __hash__(self):
        return hash((self.p, self.boxes))


def precompute_target_distances(targets, walls):
    global GLOBAL_TARGET_DISTANCES, GLOBAL_WALLS_HASH
    walls_hash = hash(frozenset(walls))

    # Skip if we have already precomputed distances for this specific map layout
    if GLOBAL_WALLS_HASH == walls_hash and len(GLOBAL_TARGET_DISTANCES) == len(targets):
        return

    GLOBAL_WALLS_HASH = walls_hash
    GLOBAL_TARGET_DISTANCES.clear()

    for t in targets:
        distances = {t: 0}
        queue = deque([t])
        while queue:
            curr = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nxt = (curr[0] + dx, curr[1] + dy)
                if nxt not in walls and nxt not in distances:
                    distances[nxt] = distances[curr] + 1
                    queue.append(nxt)
        GLOBAL_TARGET_DISTANCES[t] = distances


def get_action(state, targets, walls, algo):
    start_time = time.time()
    my_pos = state.p1
    opp_pos = state.p2

    # Treat opponent and secured boxes as solid walls
    effective_walls = walls | {opp_pos} | state.b1

    # Allow pushing of neutral boxes and opponent's secured boxes
    pushable_boxes = state.b_free | state.b2
    open_targets = targets - state.b1
    opp_targets = state.b2

    if not pushable_boxes or not open_targets:
        return "Wait"

    # Ensure distances are computed before search begins
    precompute_target_distances(targets, walls)

    init_state = LocalState(my_pos, pushable_boxes)
    init_state.blocked_action = getattr(state, "blocked_action", None)

    return search(
        init_state,
        targets,
        opp_targets,
        open_targets,
        effective_walls,
        algo,
        start_time,
    )


def get_successors(local_state, walls):
    # Actions in clockwise order
    actions = [
        ("North", (0, -1)),
        ("East", (1, 0)),
        ("South", (0, 1)),
        ("West", (-1, 0)),
    ]

    for action_name, (dx, dy) in actions:
        nx, ny = local_state.p[0] + dx, local_state.p[1] + dy
        np = (nx, ny)

        # Apply penalty if the collision engine repeatedly blocked this move
        cost = (
            100
            if hasattr(local_state, "blocked_action")
            and action_name == local_state.blocked_action
            else 1
        )

        if np not in walls and np not in local_state.boxes:
            yield (LocalState(np, local_state.boxes), action_name, cost)
        elif np in local_state.boxes:
            push_p = (nx + dx, ny + dy)
            if push_p not in walls and push_p not in local_state.boxes:
                new_boxes = set(local_state.boxes)
                new_boxes.remove(np)
                new_boxes.add(push_p)
                yield (LocalState(np, new_boxes), action_name, cost)


def heuristic(local_state, open_targets, walls):
    boxes = list(local_state.boxes)
    targets = list(open_targets)

    if len(boxes) != len(targets):
        return float("inf")

    # Look up precomputed static distances instead of running BFS per state
    dist_matrix = []
    for b in boxes:
        row = []
        for t in targets:
            d = GLOBAL_TARGET_DISTANCES.get(t, {}).get(b, float("inf"))
            row.append(d)
        dist_matrix.append(row)

    # Evaluate box-to-target permutations for minimum cost
    min_cost = float("inf")
    for perm in itertools.permutations(range(len(targets))):
        cost = sum(dist_matrix[i][perm[i]] for i in range(len(boxes)))
        if cost < min_cost:
            min_cost = cost

    return min_cost


def search(init_state, all_targets, opp_targets, open_targets, walls, algo, start_time):
    frontier = []
    initial_h = heuristic(init_state, open_targets, walls) if algo == "A*" else 0

    if initial_h == float("inf"):
        return "Wait"

    heapq.heappush(frontier, (initial_h, 0, 0, init_state, None))
    explored = set()
    counter = 1

    init_boxes_on_targets = init_state.boxes & all_targets
    init_opp_boxes = init_state.boxes & opp_targets

    best_action = "Wait"

    while frontier:
        # Halt execution at 0.95 seconds to safely return before GUI force-wait timeout
        if time.time() - start_time > 0.95:
            return best_action

        priority, cost, _, current_state, first_action = heapq.heappop(frontier)

        # Continuously track the first promising action pulled from priority queue
        if first_action and best_action == "Wait":
            best_action = first_action

        curr_boxes_on_targets = current_state.boxes & all_targets
        curr_opp_boxes = current_state.boxes & opp_targets

        newly_scored = curr_boxes_on_targets - init_boxes_on_targets
        stolen = init_opp_boxes - curr_opp_boxes

        # Stop search if an action scores or steals a box
        if newly_scored or stolen:
            return first_action if first_action else "Wait"

        if current_state not in explored:
            explored.add(current_state)

            for successor, action, step_cost in get_successors(current_state, walls):
                if successor not in explored:
                    h = heuristic(successor, open_targets, walls)
                    if h == float("inf"):
                        continue

                    new_cost = cost + step_cost
                    prio = new_cost + h if algo == "A*" else h
                    root_action = first_action if first_action else action

                    # Store immediate improvement as fallback action
                    if h < initial_h and best_action == "Wait":
                        best_action = root_action

                    heapq.heappush(
                        frontier, (prio, new_cost, counter, successor, root_action)
                    )
                    counter += 1

    return best_action
