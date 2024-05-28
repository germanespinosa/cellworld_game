import math
import pygame
import typing
from .model import Model
from .coordinate_converter import CoordinateConverter
import colorsys


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n  # Evenly distribute hues in the [0, 1) interval
        saturation = 0.7  # Choose a saturation level that avoids white and grays
        lightness = 0.5  # Choose a lightness level that avoids black and white
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert the RGB values from [0, 1] to [0, 255] and round them to integers
        rgb = tuple(round(c * 255) for c in rgb)
        colors.append(rgb)

    return colors


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
        self.render_steps: typing.List[typing.Callable[[pygame.Surface, CoordinateConverter], None]] = []
        self.render_steps_z_index: typing.List[int] = []
        self.render_steps_sequence: typing.List[int] = []
        self.add_render_step(render_step=self.render_arena, z_index=0)
        self.add_render_step(render_step=self.render_occlusions, z_index=10)
        self.add_render_step(render_step=self.render_visibility, z_index=20)
        for agent_name, agent in model.agents.items():
            agent.set_sprite_size((screen_width/10, screen_width/10))
            agent.set_coordinate_converter(self.coordinate_converter)
            self.add_render_step(agent.render, z_index=100)
        self.screen = pygame.display.set_mode(self.coordinate_converter.screen_size)
        self.arena_color = (210, 210, 210)
        self.occlusion_color = (50, 50, 50)
        self.visibility_color = (255, 255, 255)
        self.background_color = (0, 0, 0)
        self.agent_colors = {agent_name: color for agent_name, color
                             in zip(self.model.agents.keys(),
                                    generate_distinct_colors(len(self.model.agents)))}
        self.clock = pygame.time.Clock()
        self.agent_perspective = -1
        self.show_sprites = True
        self.target = None
        self.on_mouse_button_down = self.on_mouse_down_event
        self.on_mouse_button_up = None
        self.on_mouse_move = None
        self.on_mouse_wheel = None
        self.on_key_up = None
        self.on_quit = None
        self.on_frame = None
        self.pressed_keys = pygame.key.get_pressed()
        self.draw = self.render
        self.visibility_location = (.5, .5)

    def on_mouse_down_event(self, event, location):
        self.visibility_location = location

    def add_render_step(self,
                        render_step: typing.Callable[[pygame.Surface, CoordinateConverter], None],
                        z_index: int = -1):
        self.render_steps.append(render_step)
        self.render_steps_z_index.append(z_index)
        self.render_steps_sequence = [i for i, z in
                                      sorted([(i, math.inf if z == -1 else z)
                                             for i, z
                                             in enumerate(self.render_steps_z_index)],
                                             key=lambda item: item[1])]

    def render_occlusions(self, surface, coordinate_converter):
        for occlusion in self.model.occlusions:
            occlusion.render(surface=surface,
                             coordinate_converter=coordinate_converter,
                             color=self.occlusion_color)

    def render_arena(self, surface, coordinate_converter):
        self.model.arena.render(surface=surface,
                                coordinate_converter=coordinate_converter,
                                color=self.arena_color)

    def render_visibility(self, surface, coordinate_converter):
        if self.agent_perspective != -1:
            agent_name = list(self.model.agents.keys())[self.agent_perspective]
            self.visibility.render(surface=surface,
                                   coordinate_converter=coordinate_converter,
                                   location=self.model.agents[agent_name].state.location,
                                   direction=self.model.agents[agent_name].state.direction,
                                   view_field=self.model.agents[agent_name].view_field,
                                   color=self.visibility_color)

    def render(self):
        self.screen.fill(self.background_color)
        for render_step_number in self.render_steps_sequence:
            render_step = self.render_steps[render_step_number]
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
        if key == pygame.K_p:
            self.model.pause()
