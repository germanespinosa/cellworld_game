import math
import pygame
import typing
from .model import Model
from .agent import CoordinateConverter
from .util import generate_distinct_colors


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
        self.coordinate_converter = CoordinateConverter(screen_size=(screen_width, int(self.hexa_ratio * screen_width)),
                                                        flip_y=flip_y)
        for agent_name, agent in model.agents.items():
            agent.set_sprite_size((screen_width/10, screen_width/10))
            agent.set_coordinate_converter(self.coordinate_converter)
        self.screen = pygame.display.set_mode(self.coordinate_converter.screen_size)
        self.arena_color = (255, 255, 255)
        self.occlusion_color = (50, 50, 50)
        self.agent_colors = {agent_name: color for agent_name, color
                             in zip(self.model.agents.keys(),
                                    generate_distinct_colors(len(self.model.agents)))}
        self.clock = pygame.time.Clock()
        self.agent_perspective = -1
        self.show_sprites = True
        self.target = None
        self.on_mouse_button_down = None
        self.on_mouse_button_up = None
        self.on_mouse_move = None
        self.on_mouse_wheel = None
        self.on_key_up = None
        self.on_quit = None
        self.on_frame = None
        self.pressed_keys = pygame.key.get_pressed()
        self.render_steps: typing.List[typing.Callable[[pygame.Surface, CoordinateConverter], None]] = []

    def add_render_step(self,
                        render_step: typing.Callable[[pygame.Surface, CoordinateConverter], None]):
        self.render_steps.append(render_step)

    def draw_polygon(self, polygon, color):
        """Draws a hexagon at the specified position and size."""
        pygame.draw.polygon(self.screen,
                            color,
                            [self.coordinate_converter.from_canonical(point) for point in polygon.exterior.coords])

    def draw_polygon_vertices(self, polygon, color, size=2):
        """Draws a hexagon at the specified position and size."""
        for point in polygon.exterior.coords:
            pygame.draw.circle(surface=self.screen,
                               color=color,
                               center=self.coordinate_converter.from_canonical(point),
                               radius=size,
                               width=2)

    def draw_points(self, points, color, size=2):
        """Draws a hexagon at the specified position and size."""
        for point in points:
            pygame.draw.circle(surface=self.screen,
                               color=color,
                               center=self.coordinate_converter.from_canonical((point.x, point.y)),
                               radius=size,
                               width=2)

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

        for (name, agent), color in zip(self.model.agents.items(), self.agent_colors):
            if agent.visible:
                if self.show_sprites:
                    agent.draw(surface=self.screen,
                               coordinate_converter=self.coordinate_converter)
                else:
                    self.draw_polygon(self.model.agents[name].get_polygon(), color=self.agent_colors[name])

        for render_step in self.render_steps:
            render_step(self.screen, self.coordinate_converter)

        self.__process_events__()
        if self.on_frame:
            self.on_frame(self.screen, self.coordinate_converter)
        pygame.display.flip()

    def __process_events__(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.on_quit:
                    self.on_quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                if self.on_mouse_button_down:
                    self.on_mouse_button_down(event.button, canonical_x_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                if self.on_mouse_button_up:
                    self.on_mouse_button_up(event.button, canonical_x_y)
            elif event.type == pygame.MOUSEMOTION:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                if self.on_mouse_move:
                    self.on_mouse_move(canonical_x_y)
            elif event.type == pygame.MOUSEWHEEL:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                if self.on_mouse_wheel:
                    self.on_mouse_wheel(event.button, canonical_x_y)
            elif event.type == pygame.KEYDOWN:
                if self.on_key_down:
                    self.on_key_down(key=event.key)
            elif event.type == pygame.KEYUP:
                if self.on_key_up:
                    self.on_key_up(event.key)
        self.pressed_keys = pygame.key.get_pressed()

    def on_key_down(self, key):
        if key == pygame.K_0:
            self.agent_perspective = -1
        if key == pygame.K_1:
            self.agent_perspective = 0
        if key == pygame.K_2:
            self.agent_perspective = 1
        if key == pygame.K_3:
            self.show_sprites = False
        if key == pygame.K_4:
            self.show_sprites = True
