import os
import sys
import time
import pygame

# Route to requirement_7
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(source_dir, "requirement_7"))

try:
    from agent_1 import get_action as get_action_1
    from agent_2 import get_action as get_action_2
except ImportError:
    print("Warning: Agents not found. Ensure agent_1.py and agent_2.py exist.")
    def get_action_1(*args, **kwargs): return "Wait"
    def get_action_2(*args, **kwargs): return "Wait"

class CompetitiveState:
    def __init__(self, p1, p2, b_free, b1, b2, t, blocked_action=None):
        self.p1 = p1
        self.p2 = p2
        self.b_free = frozenset(b_free)
        self.b1 = frozenset(b1)
        self.b2 = frozenset(b2)
        self.boxes = self.b_free | self.b1 | self.b2
        self.t = t
        self.blocked_action = blocked_action

class CompetitiveGameLogic:
    def __init__(self, map_file_name="example_map.txt", step_limit=50):
        self.walls = set()
        self.targets = set()
        self.b_free = set()
        self.b1 = set()
        self.b2 = set()
        self.boxes = set()

        self.p1 = None
        self.p2 = None
        self.width = 0
        self.height = 0

        self.step_limit = step_limit
        self.current_step = 0
        self.is_paused = True
        self.map_file_name = map_file_name

        self.history_1 = []
        self.history_2 = []

        self.load_map(map_file_name)

    def reset_logic(self, new_step_limit):
        """Resets the arena for a new match."""
        self.walls.clear()
        self.targets.clear()
        self.b_free.clear()
        self.b1.clear()
        self.b2.clear()
        self.boxes.clear()

        self.history_1.clear()
        self.history_2.clear()

        self.step_limit = new_step_limit
        self.current_step = 0
        self.is_paused = True

        self.load_map(self.map_file_name)

    def load_map(self, file_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_name)

        with open(full_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f.readlines()]

        self.height = len(lines)
        self.width = max(len(line) for line in lines) if lines else 0

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                coord = (x, y)
                if char == "%":
                    self.walls.add(coord)
                elif char == "D":
                    self.targets.add(coord)
                elif char == "B":
                    self.b_free.add(coord)
                    self.boxes.add(coord)
                elif char == "1":
                    self.p1 = coord
                elif char == "2":
                    self.p2 = coord

    def get_current_state(self, agent_id):
        blocked = None
        if agent_id == 1 and len(self.history_1) >= 3:
            blocked = self.history_1[-1]
        elif agent_id == 2 and len(self.history_2) >= 3:
            blocked = self.history_2[-1]
        return CompetitiveState(self.p1, self.p2, self.b_free, self.b1, self.b2, self.current_step, blocked)

    def execute_simultaneous_moves(self, action1, action2):
        action_map = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0), "Wait": (0, 0)}
        dx1, dy1 = action_map.get(action1, (0, 0))
        dx2, dy2 = action_map.get(action2, (0, 0))

        np1 = (self.p1[0] + dx1, self.p1[1] + dy1)
        np2 = (self.p2[0] + dx2, self.p2[1] + dy2)

        box1_push = np1 if np1 in self.boxes else None
        box2_push = np2 if np2 in self.boxes else None

        bp1 = (np1[0] + dx1, np1[1] + dy1) if box1_push else None
        bp2 = (np2[0] + dx2, np2[1] + dy2) if box2_push else None

        valid1, valid2 = True, True

        if np1 in self.walls or (box1_push and (bp1 in self.walls or bp1 in self.boxes)):
            valid1 = False
        if np2 in self.walls or (box2_push and (bp2 in self.walls or bp2 in self.boxes)):
            valid2 = False

        if not valid1: np1, box1_push, bp1 = self.p1, None, None
        if not valid2: np2, box2_push, bp2 = self.p2, None, None

        cancel1, cancel2 = False, False

        if np1 == np2: cancel1 = cancel2 = True
        elif np1 == self.p2 and np2 == self.p1: cancel1 = cancel2 = True
        elif box1_push and box2_push and box1_push == box2_push: cancel1 = cancel2 = True
        elif box1_push and box2_push and bp1 == bp2: cancel1 = cancel2 = True

        if box1_push and bp1 == np2: cancel1 = cancel2 = True
        if box2_push and bp2 == np1: cancel1 = cancel2 = True

        if cancel1: np1, box1_push, bp1 = self.p1, None, None
        if cancel2: np2, box2_push, bp2 = self.p2, None, None

        if cancel1 and not cancel2:
            if np2 == np1 or (box2_push and bp2 == np1):
                np2, box2_push, bp2 = self.p2, None, None
        elif cancel2 and not cancel1:
            if np1 == np2 or (box1_push and bp1 == np2):
                np1, box1_push, bp1 = self.p1, None, None

        if cancel1 and action1 != "Wait": self.history_1.append(action1)
        else: self.history_1.clear()

        if cancel2 and action2 != "Wait": self.history_2.append(action2)
        else: self.history_2.clear()

        self.p1, self.p2 = np1, np2
        if box1_push: self._update_box_ownership(box1_push, bp1, 1)
        if box2_push: self._update_box_ownership(box2_push, bp2, 2)
        self.current_step += 1

    def _update_box_ownership(self, old_pos, new_pos, agent_id):
        self.boxes.remove(old_pos)
        self.boxes.add(new_pos)

        if old_pos in self.b1: self.b1.remove(old_pos)
        elif old_pos in self.b2: self.b2.remove(old_pos)
        else: self.b_free.discard(old_pos)

        if new_pos in self.targets:
            if agent_id == 1: self.b1.add(new_pos)
            else: self.b2.add(new_pos)
        else:
            self.b_free.add(new_pos)

class CompetitiveGUI:
    def __init__(self, game: CompetitiveGameLogic, tile_size: int = 64):
        self.game = game
        self.tile_size = tile_size
        pygame.init()
        self.screen_width = max(700, game.width * tile_size)
        self.screen_height = max(500, game.height * tile_size + 60)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Sokoban Multi-Agent Arena")

        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 40, bold=True)

        # State tracking
        self.state = "MENU"
        self.algo1 = "A*"
        self.algo2 = "GBFS"
        self.step_limit = 50

        # Menu Layout configuration
        cx = self.screen_width // 2
        cy = self.screen_height // 2
        self.btn_a1_astar = pygame.Rect(cx - 50, cy - 80, 70, 35)
        self.btn_a1_gbfs  = pygame.Rect(cx + 30, cy - 80, 70, 35)
        self.btn_a2_astar = pygame.Rect(cx - 50, cy - 20, 70, 35)
        self.btn_a2_gbfs  = pygame.Rect(cx + 30, cy - 20, 70, 35)
        self.btn_minus    = pygame.Rect(cx - 50, cy + 40, 35, 35)
        self.btn_plus     = pygame.Rect(cx + 65, cy + 40, 35, 35)
        self.btn_start    = pygame.Rect(cx - 100, cy + 110, 200, 50)

    def draw_tile(self, x, y, color, is_circle=False, padding=0):
        rect = (x * self.tile_size + padding, y * self.tile_size + padding,
                self.tile_size - padding * 2, self.tile_size - padding * 2)
        if is_circle:
            center = (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2)
            pygame.draw.circle(self.screen, color, center, self.tile_size // 4)
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=6)

    def render_menu(self):
        self.screen.fill((40, 45, 55))

        title_surf = self.title_font.render("Multi-Agent Arena Configuration", True, (255, 255, 255))
        self.screen.blit(title_surf, title_surf.get_rect(center=(self.screen_width // 2, 80)))

        # Labels
        lbl_p1 = self.font.render("Agent 1 (Blue):", True, (100, 180, 255))
        lbl_p2 = self.font.render("Agent 2 (Red):", True, (255, 100, 100))
        lbl_steps = self.font.render("Step Limit:", True, (255, 255, 255))

        cx = self.screen_width // 2
        cy = self.screen_height // 2
        self.screen.blit(lbl_p1, (cx - 220, cy - 75))
        self.screen.blit(lbl_p2, (cx - 220, cy - 15))
        self.screen.blit(lbl_steps, (cx - 220, cy + 45))

        # Helper for drawing toggle buttons
        def draw_btn(rect, text, is_active, color_active):
            color = color_active if is_active else (80, 80, 80)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            t_surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(t_surf, t_surf.get_rect(center=rect.center))

        draw_btn(self.btn_a1_astar, "A*", self.algo1 == "A*", (50, 120, 255))
        draw_btn(self.btn_a1_gbfs, "GBFS", self.algo1 == "GBFS", (50, 120, 255))

        draw_btn(self.btn_a2_astar, "A*", self.algo2 == "A*", (220, 50, 50))
        draw_btn(self.btn_a2_gbfs, "GBFS", self.algo2 == "GBFS", (220, 50, 50))

        draw_btn(self.btn_minus, "-", False, (100, 100, 100))
        draw_btn(self.btn_plus, "+", False, (100, 100, 100))

        steps_surf = self.font.render(str(self.step_limit), True, (255, 255, 255))
        self.screen.blit(steps_surf, steps_surf.get_rect(center=(cx + 25, cy + 57)))

        # Start button
        mouse_pos = pygame.mouse.get_pos()
        start_color = (80, 200, 120) if self.btn_start.collidepoint(mouse_pos) else (50, 160, 90)
        pygame.draw.rect(self.screen, start_color, self.btn_start, border_radius=8)
        start_t = self.font.render("START MATCH", True, (255, 255, 255))
        self.screen.blit(start_t, start_t.get_rect(center=self.btn_start.center))

        pygame.display.flip()

    def render_playing(self):
        self.screen.fill((40, 40, 40))

        for x, y in self.game.walls: self.draw_tile(x, y, (100, 100, 100))
        for x, y in self.game.targets: self.draw_tile(x, y, (240, 200, 80), is_circle=True)
        for x, y in self.game.b_free: self.draw_tile(x, y, (180, 120, 70), padding=8)
        for x, y in self.game.b1: self.draw_tile(x, y, (100, 180, 255), padding=8)
        for x, y in self.game.b2: self.draw_tile(x, y, (255, 100, 100), padding=8)

        self.draw_tile(self.game.p1[0], self.game.p1[1], (50, 120, 255), padding=12)
        self.draw_tile(self.game.p2[0], self.game.p2[1], (255, 50, 50), padding=12)

        info = f"Step: {self.game.current_step}/{self.game.step_limit} | "
        info += f"P1: {len(self.game.b1)} | P2: {len(self.game.b2)}"

        status_color = (255, 255, 255)
        if self.game.current_step >= self.game.step_limit:
            if len(self.game.b1) > len(self.game.b2): info += " [BLUE WINS]"
            elif len(self.game.b2) > len(self.game.b1): info += " [RED WINS]"
            else: info += " [DRAW]"
            status_color = (100, 255, 100)
        elif self.game.is_paused:
            info += " [PAUSED]"

        info += " | ESC: Menu"
        text = self.font.render(info, True, status_color)
        self.screen.blit(text, (15, self.game.height * self.tile_size + 15))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        auto_timer = 0

        while running:
            dt = clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.state == "MENU":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        pos = event.pos
                        if self.btn_a1_astar.collidepoint(pos): self.algo1 = "A*"
                        elif self.btn_a1_gbfs.collidepoint(pos): self.algo1 = "GBFS"
                        elif self.btn_a2_astar.collidepoint(pos): self.algo2 = "A*"
                        elif self.btn_a2_gbfs.collidepoint(pos): self.algo2 = "GBFS"
                        elif self.btn_minus.collidepoint(pos): self.step_limit = max(10, self.step_limit - 10)
                        elif self.btn_plus.collidepoint(pos): self.step_limit = min(500, self.step_limit + 10)
                        elif self.btn_start.collidepoint(pos):
                            self.game.reset_logic(self.step_limit)
                            self.state = "PLAYING"

                elif self.state == "PLAYING":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.game.is_paused = not self.game.is_paused
                        elif event.key == pygame.K_ESCAPE:
                            self.state = "MENU"

            if self.state == "PLAYING" and not self.game.is_paused and self.game.current_step < self.game.step_limit:
                auto_timer += dt
                if auto_timer >= 500:
                    state1 = self.game.get_current_state(1)
                    state2 = self.game.get_current_state(2)

                    t0 = time.time()
                    act1 = get_action_1(state1, self.game.targets, self.game.walls, self.algo1)
                    if time.time() - t0 > 1.0: act1 = "Wait"

                    t1 = time.time()
                    act2 = get_action_2(state2, self.game.targets, self.game.walls, self.algo2)
                    if time.time() - t1 > 1.0: act2 = "Wait"

                    self.game.execute_simultaneous_moves(act1, act2)
                    auto_timer = 0

            if self.state == "MENU":
                self.render_menu()
            elif self.state == "PLAYING":
                self.render_playing()

        pygame.quit()

if __name__ == "__main__":
    game_instance = CompetitiveGameLogic("example_map.txt")
    gui = CompetitiveGUI(game_instance)
    gui.run()
