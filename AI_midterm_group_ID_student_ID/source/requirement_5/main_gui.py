import os
import sys

import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(source_dir, "requirement_2"))
sys.path.append(os.path.join(source_dir, "requirement_4"))

try:
    from algorithms import SokobanState
except ImportError:

    class SokobanState:
        def __init__(self, agent_pos, boxes):
            self.agent_pos = agent_pos
            self.boxes = frozenset(boxes)

        def __eq__(self, other):
            return (
                isinstance(other, SokobanState)
                and self.agent_pos == other.agent_pos
                and self.boxes == other.boxes
            )

        def __hash__(self):
            return hash((self.agent_pos, self.boxes))


class Agent:
    def __init__(self, agent_id: int, x: int, y: int, color=(50, 150, 255)):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.color = color

    @property
    def pos(self):
        return (self.x, self.y)

    @pos.setter
    def pos(self, position):
        self.x, self.y = position

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy


class GameLogic:
    def __init__(self, map_file_name="example_map.txt"):
        self.walls = set()
        self.targets = set()
        self.boxes = set()
        self.agents = []
        self.width = 0
        self.height = 0

        self.is_paused = True
        self.history = []
        self.history_index = 0

        self.load_map(map_file_name)

    @property
    def initial_agent_pos(self):
        return self.agents[0].pos if self.agents else None

    @property
    def initial_boxes(self):
        return self.history[0][1] if self.history else set()

    def load_map(self, file_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_name)

        if not os.path.exists(full_path):
            print(f"Lỗi: Không tìm thấy file bản đồ tại đường dẫn: {full_path}")
            sys.exit(1)

        with open(full_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f.readlines()]

        self.height = len(lines)
        self.width = max(len(line) for line in lines) if lines else 0

        color_palette = [(50, 150, 255), (255, 100, 100), (100, 255, 100)]
        agent_counter = 0

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == "%":
                    self.walls.add((x, y))
                elif char == "D":
                    self.targets.add((x, y))
                elif char == "B":
                    self.boxes.add((x, y))
                elif char == "A":
                    color = color_palette[agent_counter % len(color_palette)]
                    self.agents.append(Agent(agent_counter, x, y, color=color))
                    agent_counter += 1
                elif char == "C":
                    self.boxes.add((x, y))
                    self.targets.add((x, y))

        if not self.agents:
            print("Lỗi: Không tìm thấy vị trí Agent ('A') trong file map!")
            sys.exit(1)

        initial_positions = [a.pos for a in self.agents]
        self.history = [(initial_positions, tuple(self.boxes))]
        self.history_index = 0

    def execute_move(self, dx: int, dy: int, agent_index: int = 0):
        if agent_index >= len(self.agents):
            return False

        curr_agent = self.agents[agent_index]
        curr_x, curr_y = curr_agent.pos
        new_x, new_y = curr_x + dx, curr_y + dy

        if (new_x, new_y) in self.walls:
            return False

        new_boxes = set(self.boxes)
        if (new_x, new_y) in new_boxes:
            box_next_x, box_next_y = new_x + dx, new_y + dy
            if (box_next_x, box_next_y) in self.walls or (
                box_next_x,
                box_next_y,
            ) in new_boxes:
                return False
            new_boxes.remove((new_x, new_y))
            new_boxes.add((box_next_x, box_next_y))

        curr_agent.pos = (new_x, new_y)
        self.boxes = new_boxes

        self.history = self.history[: self.history_index + 1]
        agent_positions = [a.pos for a in self.agents]
        self.history.append((agent_positions, tuple(self.boxes)))
        self.history_index += 1
        return True

    def move_backward(self):
        if self.history_index > 0:
            self.history_index -= 1
            positions, boxes = self.history[self.history_index]
            for idx, pos in enumerate(positions):
                if idx < len(self.agents):
                    self.agents[idx].pos = pos
            self.boxes = set(boxes)

    def move_forward(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            positions, boxes = self.history[self.history_index]
            for idx, pos in enumerate(positions):
                if idx < len(self.agents):
                    self.agents[idx].pos = pos
            self.boxes = set(boxes)

    def toggle_pause(self):
        self.is_paused = not self.is_paused


class SokobanGUI:
    def __init__(self, game: GameLogic, tile_size: int = 64):
        self.game = game
        self.tile_size = tile_size
        pygame.init()
        self.screen = pygame.display.set_mode(
            (game.width * tile_size, game.height * tile_size + 40)
        )
        pygame.display.set_caption("Sokoban AI Visualizer")
        self.font = pygame.font.SysFont("Arial", 20)

    def draw_wall(self, x: int, y: int):
        rect = (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, (100, 100, 100), rect)

    def draw_goal(self, x: int, y: int):
        center = (
            x * self.tile_size + self.tile_size // 2,
            y * self.tile_size + self.tile_size // 2,
        )
        pygame.draw.circle(self.screen, (240, 200, 80), center, self.tile_size // 5)

    def draw_box(self, x: int, y: int):
        color = (80, 200, 120) if (x, y) in self.game.targets else (180, 100, 50)
        rect = (
            x * self.tile_size + 6,
            y * self.tile_size + 6,
            self.tile_size - 12,
            self.tile_size - 12,
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def draw_agent(self, agent: Agent):
        ax, ay = agent.pos
        rect = (
            ax * self.tile_size + 10,
            ay * self.tile_size + 10,
            self.tile_size - 20,
            self.tile_size - 20,
        )
        pygame.draw.rect(self.screen, agent.color, rect, border_radius=8)

    def render(self):
        self.screen.fill((30, 30, 30))

        for x, y in self.game.walls:
            self.draw_wall(x, y)

        for x, y in self.game.targets:
            self.draw_goal(x, y)

        for x, y in self.game.boxes:
            self.draw_box(x, y)

        for agent in self.game.agents:
            self.draw_agent(agent)

        status_text = f"Step: {self.game.history_index}/{len(self.game.history) - 1} | "
        status_text += "STATUS: PAUSED" if self.game.is_paused else "STATUS: ACTIVE"
        text_surface = self.font.render(status_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, self.game.height * self.tile_size + 8))

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True

        auto_play_timer = 0
        delay_between_steps = 300

        while running:
            dt = clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.game.toggle_pause()
                    elif event.key == pygame.K_LEFT:
                        self.game.move_backward()
                    elif event.key == pygame.K_RIGHT:
                        self.game.move_forward()

            if not self.game.is_paused:
                auto_play_timer += dt
                if auto_play_timer >= delay_between_steps:
                    self.game.move_forward()
                    auto_play_timer = 0

            self.render()

        pygame.quit()


if __name__ == "__main__":
    import time

    from a_star import a_star_search
    from algorithms import uniform_cost_search
    from heuristic import precompute_maze_distances

    game = GameLogic("example_map.txt")
    init_state = SokobanState(game.initial_agent_pos, game.initial_boxes)

    print("--- Sokoban AI ---")
    print("1: Uniform Cost Search (UCS)")
    print("2: A* Search")
    choice = input("Select algorithm (1 or 2): ")

    solution_path = None
    start_time = time.time()

    if choice == "1":
        print("Calculating UCS...")
        solution_path, _, _ = uniform_cost_search(init_state, game.targets, game.walls)
    elif choice == "2":
        print("Calculating A*...")
        dist_table = precompute_maze_distances(
            game.targets, game.walls, game.width, game.height
        )
        solution_path, _, _ = a_star_search(
            init_state, game.targets, game.walls, dist_table
        )

    calc_duration = time.time() - start_time
    if not solution_path:
        print("No solution found by AI.")
        sys.exit()

    print(f"Solution found in {len(solution_path)} steps.")
    print(f"Calculation time: {calc_duration:.4f} seconds.")
    print("Launching GUI... (Press SPACE in the GUI window to start)")

    action_map = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0)}
    for action in solution_path:
        dx, dy = action_map[action]
        game.execute_move(dx, dy)

    while game.history_index > 0:
        game.move_backward()

    gui = SokobanGUI(game)
    gui.run()
