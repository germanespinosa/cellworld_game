import math
import pygame
import typing
from .model import Model
from .coordinate_converter import CoordinateConverter
import colorsys
from .event import EventDispatcher


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


class View(EventDispatcher):
    def __init__(self,
                 model: Model,
                 screen_width: int = 800,
                 flip_y: bool = True):
        pygame.init()
        self.model = model
        self.hexa_ratio = (math.sqrt(3) / 2)
        self.coordinate_converter = CoordinateConverter(screen_size=(screen_width, int(self.hexa_ratio * screen_width)),
                                                        flip_y=flip_y)
        self.render_steps: typing.List[typing.Callable[[pygame.Surface, CoordinateConverter], None]] = []
        self.render_steps_z_index: typing.List[int] = []
        self.render_steps_sequence: typing.List[int] = []
        self.screen = pygame.display.set_mode(self.coordinate_converter.screen_size)
        self.background_color = (0, 0, 0)
        self.clock = pygame.time.Clock()
        self.show_sprites = True
        EventDispatcher.__init__(self, ["mouse_button_down",
                                        "mouse_button_up",
                                        "mouse_move",
                                        "mouse_wheel",
                                        "key_down",
                                        "key_up",
                                        "quit",
                                        "frame"])
        self.pressed_keys = pygame.key.get_pressed()
        self.draw = self.render
        self.visibility_location = (.5, .5)

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

    def render(self):
        self.screen.fill(self.background_color)
        for render_step_number in self.render_steps_sequence:
            render_step = self.render_steps[render_step_number]
            render_step(self.screen, self.coordinate_converter)
        self.__process_events__()
        self.__dispatch__("frame", self.screen, self.coordinate_converter)
        pygame.display.flip()

    def __process_events__(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__dispatch__("quit")
            if event.type == pygame.MOUSEBUTTONDOWN:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                self.__dispatch__("mouse_button_down", canonical_x_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                self.__dispatch__("mouse_button_up", canonical_x_y)
            elif event.type == pygame.MOUSEMOTION:
                canonical_x_y = self.coordinate_converter.to_canonical(event.pos)
                self.__dispatch__("mouse_move", canonical_x_y)
            elif event.type == pygame.MOUSEWHEEL:
                canonical_x_y = self.coordinate_converter.to_canonical((event.x, event.y))
                self.__dispatch__("mouse_wheel", canonical_x_y)
            elif event.type == pygame.KEYDOWN:
                self.__dispatch__("key_down", event.key)
            elif event.type == pygame.KEYUP:
                self.__dispatch__("key_up", event.key)
        self.pressed_keys = pygame.key.get_pressed()

    def on_key_down(self, key):
        if key == pygame.K_TAB:
            self.show_sprites = not self.show_sprites