from collections import deque

def precompute_maze_distances(targets, walls, width, height):
    """
    Runs a backward BFS from every designated target cell to all walkable cells.
    Returns a dictionary: dist_table[target_coord][cell_coord] = shortest_steps.
    Obstacle-aware maze distance avoids Euclidean/Manhattan formulas entirely.
    """
    dist_table = {}
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]

    for target in targets:
        dist_table[target] = {}
        queue = deque([(target, 0)])
        visited = {target}

        while queue:
            current, steps = queue.popleft()
            dist_table[target][current] = steps

            for dx, dy in directions:
                nxt = (current[0] + dx, current[1] + dy)
                if (0 <= nxt[0] < width and 0 <= nxt[1] < height
                        and nxt not in walls and nxt not in visited):
                    visited.add(nxt)
                    queue.append((nxt, steps + 1))

    return dist_table

def hungarian_min_cost(cost_matrix):
    """
    Solves the minimum-weight bipartite matching problem in O(N^3).
    A pure Python implementation of the Hungarian / Kuhn-Munkres algorithm.
    """
    n = len(cost_matrix)
    if n == 0:
        return 0

    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    return -v[0]

def is_deadlock(box, walls, targets):
    """
    Simple static corner deadlock detector:
    A box in a non-target corner formed by two perpendicular walls can never move.
    """
    if box in targets:
        return False

    x, y = box
    north = (x, y - 1) in walls
    south = (x, y + 1) in walls
    east = (x + 1, y) in walls
    west = (x - 1, y) in walls

    if (north and east) or (north and west) or (south and east) or (south and west):
        return True
    return False

def bipartite_matching_heuristic(state, targets, walls, dist_table):
    """
    Admissible and consistent heuristic:
    Calculates the minimum cost perfect matching between box positions and target cells.
    """
    boxes = list(state.B)
    targets_list = list(targets)
    n = len(boxes)

    cost_matrix = []
    for box in boxes:
        # Check corner deadlock
        if is_deadlock(box, walls, targets):
            return float('inf')

        row = []
        for target in targets_list:
            dist = dist_table.get(target, {}).get(box, float('inf'))
            if dist == float('inf'):
                return float('inf')
            row.append(dist)
        cost_matrix.append(row)

    return hungarian_min_cost(cost_matrix)
