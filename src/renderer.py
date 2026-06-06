import pygame
import numpy as np
from typing import Any

COLORS = {
    "road": (30, 30, 35),
    "lane_line": (200, 200, 180),
    "lane_dashed": (120, 120, 100),
    "ego": (50, 220, 100),
    "traffic": (220, 80, 80),
    "background": (15, 15, 20),
    "hud_bg": (0, 0, 0, 160),
    "hud_text": (255, 255, 255),
    "hud_accent": (50, 220, 100),
    "headlight": (255, 240, 150),
    "marking": (255, 255, 255),
}

SCREEN_W, SCREEN_H = 1200, 400
LANE_WIDTH = 60
LANE_COUNT = 4
ROAD_TOP = (SCREEN_H - LANE_WIDTH * LANE_COUNT) // 2
ROAD_BOTTOM = ROAD_TOP + LANE_WIDTH * LANE_COUNT

SIM_WINDOW = 120.0
SCALE_X = SCREEN_W / SIM_WINDOW


def world_to_screen(x: float, y: float, ego_x: float) -> tuple[int, int]:
    """Simülasyon koordinatlarını ekran koordinatlarına çevirir."""
    sx = int((x - ego_x + SIM_WINDOW * 0.3) * SCALE_X)
    sy = int(ROAD_TOP + (y / 4.0) * LANE_WIDTH + LANE_WIDTH // 2)
    return sx, sy


def draw_road(surface: pygame.Surface) -> None:
    """Yol, şeritler ve çizgileri çizer."""
    surface.fill(COLORS["background"])

    pygame.draw.rect(surface, COLORS["road"],
                     (0, ROAD_TOP, SCREEN_W, ROAD_BOTTOM - ROAD_TOP))

    pygame.draw.line(surface, COLORS["marking"],
                     (0, ROAD_TOP), (SCREEN_W, ROAD_TOP), 3)
    pygame.draw.line(surface, COLORS["marking"],
                     (0, ROAD_BOTTOM), (SCREEN_W, ROAD_BOTTOM), 3)

    for lane in range(1, LANE_COUNT):
        y = ROAD_TOP + lane * LANE_WIDTH
        dash_len, gap_len = 40, 30
        x = 0
        while x < SCREEN_W:
            pygame.draw.line(surface, COLORS["lane_dashed"],
                             (x, y), (min(x + dash_len, SCREEN_W), y), 2)
            x += dash_len + gap_len


def draw_vehicle(surface: pygame.Surface, sx: int, sy: int,
                 is_ego: bool, speed: float) -> None:
    """Tek bir aracı çizer."""
    w, h = 52, 26
    color = COLORS["ego"] if is_ego else COLORS["traffic"]

    rect = pygame.Rect(sx - w // 2, sy - h // 2, w, h)
    pygame.draw.rect(surface, color, rect, border_radius=6)

    glass_rect = pygame.Rect(sx - w // 4, sy - h // 2 + 4, w // 2, h - 8)
    glass_color = (180, 230, 255) if is_ego else (150, 150, 180)
    pygame.draw.rect(surface, glass_color, glass_rect, border_radius=3)

    far_color = COLORS["headlight"]
    pygame.draw.circle(surface, far_color, (sx + w // 2 - 4, sy - 7), 4)
    pygame.draw.circle(surface, far_color, (sx + w // 2 - 4, sy + 7), 4)

    if is_ego:
        bar_max = 80
        bar_w = int((speed / bar_max) * 50)
        pygame.draw.rect(surface, (30, 30, 30),
                         (sx - 25, sy + h // 2 + 4, 50, 6), border_radius=3)
        pygame.draw.rect(surface, COLORS["hud_accent"],
                         (sx - 25, sy + h // 2 + 4, bar_w, 6), border_radius=3)


def draw_hud(surface: pygame.Surface, font: pygame.font.Font,
             speed: float, reward: float, episode: int, step: int) -> None:
    """HUD (heads-up display) çizer."""
    lines = [
        ("EPISODE", f"{episode:03d}", COLORS["hud_accent"]),
        ("STEP", f"{step:04d}", COLORS["hud_text"]),
        ("SPEED", f"{speed * 3.6:.1f} km/h", COLORS["hud_accent"]),
        ("REWARD", f"{reward:.2f}", COLORS["hud_accent"]),
    ]

    panel_x, panel_y = 20, 20
    for label, value, color in lines:
        label_surf = font.render(label, True, (150, 150, 150))
        value_surf = font.render(value, True, color)
        surface.blit(label_surf, (panel_x, panel_y))
        surface.blit(value_surf, (panel_x + 90, panel_y))
        panel_y += 28

    tag = font.render("PPO · highway-v0", True, (100, 100, 120))
    surface.blit(tag, (SCREEN_W - tag.get_width() - 20, 20))


class CustomRenderer:
    """Highway-env için özel Pygame renderer."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Highway RL – Custom Renderer")
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        self.clock = pygame.time.Clock()

    def render(self, env: Any, reward: float,
               episode: int, step: int) -> np.ndarray:
        """Bir frame render eder ve numpy array olarak döner."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        draw_road(self.screen)

        ego = env.unwrapped.vehicle
        ego_x = ego.position[0]
        ego_speed = ego.speed

        for v in env.unwrapped.road.vehicles:
            sx, sy = world_to_screen(v.position[0], v.position[1], ego_x)
            if -60 < sx < SCREEN_W + 60:
                is_ego = (v is ego)
                draw_vehicle(self.screen, sx, sy, is_ego, v.speed)

        draw_hud(self.screen, self.font, ego_speed, reward, episode, step)

        pygame.display.flip()
        self.clock.tick(60)

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        pygame.quit()