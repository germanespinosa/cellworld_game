import pygame
from .model import Model
from .agent import Agent
import math
from .util import generate_distinct_colors
import shapely as sp


class View(object):
    def __init__(self,
                 model: Model,
                 screen_width: int = 800,
                 flip_y: bool = True):
        pygame.init()
        self.model = model
        self.visibility = model.visibility
        self.screen_width = screen_width
        self.hexa_ratio = (math.sqrt(3) / 2)
        self.screen_height = int(self.hexa_ratio * screen_width)
        self.flip_y = flip_y
        self.screen_offset = (self.screen_width - self.screen_height) / 2
        self.screen_size = (screen_width, self.screen_height)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.arena_color = (255, 255, 255)
        self.occlusion_color = (50, 50, 50)
        self.agent_colors = {agent_name: color for agent_name, color
                                               in zip(self.model.agents.keys(),
                                                      generate_distinct_colors(len(self.model.agents)))}
        self.clock = pygame.time.Clock()
        self.agent_perspective = -1
        self.show_sprites = True
        self.target = None
        self.screen_target = None

    def from_canonical(self, canonical: tuple):
        canonical_x, canonical_y = canonical
        screen_x = canonical_x * self.screen_width
        if self.flip_y:
            screen_y = (1-canonical_y) * self.screen_width - self.screen_offset
        else:
            screen_y = canonical_y * self.screen_width - self.screen_offset
        return screen_x, screen_y

    def to_canonical(self, screen_x: int, screen_y: int):
        y = self.screen_height - screen_y + self.screen_offset
        canonical_y = y / self.screen_height * self.hexa_ratio
        canonical_x = screen_x / self.screen_width
        return canonical_x, canonical_y

    def draw_polygon(self, polygon, color):
        """Draws a hexagon at the specified position and size."""
        pygame.draw.polygon(self.screen,
                            color,
                            [self.from_canonical(point) for point in polygon.exterior.coords])

    def draw_polygon_vertices(self, polygon, color, size=2):
        """Draws a hexagon at the specified position and size."""
        for point in polygon.exterior.coords:
            pygame.draw.circle(surface=self.screen,
                               color=color,
                               center=self.from_canonical(point),
                               radius=size,
                               width=2)

    def draw_points(self, points, color, size=2):
        """Draws a hexagon at the specified position and size."""
        for point in points:
            pygame.draw.circle(surface=self.screen,
                               color=color,
                               center=self.from_canonical((point.x, point.y)),
                               radius=size,
                               width=2)

    def draw_agent(self, agent: Agent):
        agent_sprite: pygame.Surface = agent.get_sprite()
        width, height = agent_sprite.get_size()
        screen_x, screen_y = self.from_canonical(agent.state.location)
        self.screen.blit(agent_sprite, (screen_x - width / 2, screen_y - height / 2))

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_polygon(self.model.arena, self.arena_color)

        if self.agent_perspective != -1:
            agent_name = list(self.model.agents.keys())[self.agent_perspective]
            visibility_perspective = self.model.agents[agent_name].state
            visibility_polygon, a = self.visibility.get_visibility_polygon(location=visibility_perspective.location,
                                                                           direction=visibility_perspective.direction,
                                                                           view_field=360)
            self.draw_polygon(visibility_polygon, (180, 180, 180))

        for occlusion in self.model.occlusions:
            self.draw_polygon(occlusion, self.occlusion_color)

        if "predator" in self.model.agents and self.model.agents["predator"].path:
            current_point = self.model.agents["predator"].state.location
            for step in self.model.agents["predator"].path:
                pygame.draw.line(self.screen,
                                 (255, 0, 0),
                                 self.from_canonical(current_point),
                                 self.from_canonical(step),
                                 2)
                pygame.draw.circle(surface=self.screen,
                                   color=(0, 0, 255),
                                   center=self.from_canonical(step),
                                   radius=5,
                                   width=2)
                current_point = step
            if "prey" in self.model.agents:
                predator_location = self.from_canonical(self.model.agents["predator"].state.location)
                puff_area_size = self.model.agents["prey"].puff_threshold * self.screen_width
                puff_location = predator_location[0] - puff_area_size , predator_location[1] - puff_area_size
                puff_area_surface = pygame.Surface((puff_area_size * 2, puff_area_size * 2), pygame.SRCALPHA)
                puff_area_color = (255, 0, 0, 60) if self.model.agents["prey"].puff_cool_down > 0 else (0, 0, 255, 60)
                pygame.draw.circle(puff_area_surface,
                                   color=puff_area_color,
                                   center=(puff_area_size, puff_area_size),
                                   radius=puff_area_size)
                self.screen.blit(puff_area_surface,
                                 puff_location)
                pygame.draw.circle(surface=self.screen,
                                   color=(0, 0, 255),
                                   center=predator_location,
                                   radius=puff_area_size,
                                   width=2)


        for (name, agent), color in zip(self.model.agents.items(), self.agent_colors):
            if self.show_sprites:
                self.draw_agent(agent=agent)
            else:
                self.draw_polygon(self.model.agents[name].get_polygon(), color=self.agent_colors[name])

        if self.screen_target:
            pygame.draw.circle(surface=self.screen,
                               color=(255, 0, 255),
                               center=self.screen_target,
                               radius=3,
                               width=2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 is the left mouse button
                    x, y = event.pos
                    self.target = self.to_canonical(x, y)
                    self.screen_target = (x, y)
                    self.model.agents["predator"].set_destination(self.target)

        # Key press handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_0]:
            self.agent_perspective = -1
        if keys[pygame.K_1]:
            self.agent_perspective = 0
        if keys[pygame.K_2]:
            self.agent_perspective = 1
        if keys[pygame.K_3]:
            self.show_sprites = False
        if keys[pygame.K_4]:
            self.show_sprites = True
        # if keys[pygame.K_w]:
        #     self.model.agents["prey"].dynamics.forward_speed = .005
        # elif keys[pygame.K_s]:
        #     self.model.agents["prey"].dynamics.forward_speed = -.005
        # else:
        #     self.model.agents["prey"].dynamics.forward_speed = 0
        # if keys[pygame.K_a]:
        #     self.model.agents["prey"].dynamics.turn_speed = 2
        # elif keys[pygame.K_d]:
        #     self.model.agents["prey"].dynamics.turn_speed = -2
        # else:
        #     self.model.agents["prey"].dynamics.turn_speed = 0
        pygame.display.flip()

