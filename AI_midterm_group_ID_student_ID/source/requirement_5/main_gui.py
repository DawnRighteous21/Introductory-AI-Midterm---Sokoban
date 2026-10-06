import os
import sys
import time
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(source_dir, "requirement_2"))
sys.path.append(os.path.join(source_dir, "requirement_4"))

try:
    from algorithms import a_star_search, uniform_cost_search, SokobanState
    from heuristic import precompute_maze_distances
except ImportError:
    print("Warning: Algorithm modules not found. Using dummy state class.")
    class SokobanState:
        def __init__(self, agent_pos, boxes):
            self.agent_pos = agent_pos
            self.boxes = frozenset(boxes)
        def __eq__(self, other):
            return isinstance(other, SokobanState) and self.agent_pos == other.agent_pos and self.boxes == other.boxes
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
        self.map_file_name = map_file_name

        self.load_map(map_file_name)

    @property
    def initial_agent_pos(self):
        return self.agents[0].pos if self.agents else None

    @property
    def initial_boxes(self):
        return self.history[0][1] if self.history else set()

    def reset_logic(self):
        """Resets the game state by reloading the map from scratch."""
        self.walls.clear()
        self.targets.clear()
        self.boxes.clear()
        self.agents.clear()
        self.history.clear()
        self.history_index = 0
        self.is_paused = True
        self.load_map(self.map_file_name)

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
            if (box_next_x, box_next_y) in self.walls or (box_next_x, box_next_y) in new_boxes:
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
        self.screen_width = max(600, game.width * tile_size)
        self.screen_height = max(400, game.height * tile_size + 40)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Sokoban AI Visualizer")
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 40, bold=True)

        self.state = "MENU"
        self.solve_stats = ""

        btn_w, btn_h = 300, 50
        cx = self.screen_width // 2 - btn_w // 2
        cy = self.screen_height // 2
        self.buttons = [
            pygame.Rect(cx, cy - 30, btn_w, btn_h),
            pygame.Rect(cx, cy + 40, btn_w, btn_h)
        ]

    def draw_wall(self, x: int, y: int):
        rect = (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, (100, 100, 100), rect)

    def draw_goal(self, x: int, y: int):
        center = (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2)
        pygame.draw.circle(self.screen, (240, 200, 80), center, self.tile_size // 5)

    def draw_box(self, x: int, y: int):
        color = (80, 200, 120) if (x, y) in self.game.targets else (180, 100, 50)
        rect = (x * self.tile_size + 6, y * self.tile_size + 6, self.tile_size - 12, self.tile_size - 12)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def draw_agent(self, agent: Agent):
        ax, ay = agent.pos
        rect = (ax * self.tile_size + 10, ay * self.tile_size + 10, self.tile_size - 20, self.tile_size - 20)
        pygame.draw.rect(self.screen, agent.color, rect, border_radius=8)

    def render_menu(self):
        self.screen.fill((40, 45, 55))

        title_surf = self.title_font.render("Sokoban AI Solver", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 3 - 40))
        self.screen.blit(title_surf, title_rect)

        mouse_pos = pygame.mouse.get_pos()

        for i, rect in enumerate(self.buttons):
            color = (80, 160, 220) if rect.collidepoint(mouse_pos) else (60, 120, 180)
            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            text = "Run Uniform Cost Search (UCS)" if i == 0 else "Run A* Search"
            text_surf = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)

        pygame.display.flip()

    def render_loading(self):
        self.screen.fill((30, 30, 30))
        text_surf = self.title_font.render("Calculating Solution...", True, (255, 200, 50))
        text_rect = text_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text_surf, text_rect)
        pygame.display.flip()

    def render_playing(self):
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
        status_text += "STATUS: PAUSED | " if self.game.is_paused else "STATUS: ACTIVE | "
        status_text += f"ESC: Back to Menu | {self.solve_stats}"

        text_surface = self.font.render(status_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, self.game.height * self.tile_size + 8))

        pygame.display.flip()

    def solve_maze(self, algo_type):
        """Executes the selected AI logic and buffers it into the history array."""
        self.state = "LOADING"
        self.render_loading()

        self.game.reset_logic()
        init_state = SokobanState(self.game.initial_agent_pos, self.game.initial_boxes)

        start_time = time.time()
        solution_path = None

        try:
            if algo_type == "UCS":
                solution_path, _, _ = uniform_cost_search(init_state, self.game.targets, self.game.walls)
            elif algo_type == "A*":
                dist_table = precompute_maze_distances(self.game.targets, self.game.walls, self.game.width, self.game.height)
                solution_path, _, _ = a_star_search(init_state, self.game.targets, self.game.walls, dist_table)
        except Exception as e:
            print(f"Algorithm Execution Failed: {e}")
            self.state = "MENU"
            return

        calc_duration = time.time() - start_time

        if not solution_path:
            self.solve_stats = "No solution found."
        else:
            self.solve_stats = f"Solved in {calc_duration:.2f}s"
            action_map = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0)}
            for action in solution_path:
                dx, dy = action_map[action]
                self.game.execute_move(dx, dy)

            while self.game.history_index > 0:
                self.game.move_backward()

        self.state = "PLAYING"

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

                if self.state == "MENU":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mouse_pos = event.pos
                        if self.buttons[0].collidepoint(mouse_pos):
                            self.solve_maze("UCS")
                        elif self.buttons[1].collidepoint(mouse_pos):
                            self.solve_maze("A*")

                elif self.state == "PLAYING":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.game.toggle_pause()
                        elif event.key == pygame.K_LEFT:
                            self.game.move_backward()
                        elif event.key == pygame.K_RIGHT:
                            self.game.move_forward()
                        elif event.key == pygame.K_ESCAPE:
                            self.state = "MENU"

            if self.state == "PLAYING" and not self.game.is_paused:
                auto_play_timer += dt
                if auto_play_timer >= delay_between_steps:
                    self.game.move_forward()
                    auto_play_timer = 0

            if self.state == "MENU":
                self.render_menu()
            elif self.state == "PLAYING":
                self.render_playing()

        pygame.quit()

if __name__ == "__main__":
    game_instance = GameLogic("example_map.txt")
    gui = SokobanGUI(game_instance)
    gui.run()
