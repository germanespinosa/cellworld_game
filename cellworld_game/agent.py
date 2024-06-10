import pygame
import typing
from .resources import Resources
from .util import Point
from .coordinate_converter import CoordinateConverter
from .polygon import Polygon


class AgentState(object):
    def __init__(self, location: Point.type = (0, 0), direction: float = 0):
        self.location = location
        self.direction = direction

    def __iter__(self):
        yield self.location
        yield self.direction

    def update(self,
               distance: float,
               rotation: float) -> "AgentState":
        new_direction = self.direction + rotation
        return AgentState(location=Point.move(start=self.location,
                                              direction_degrees=new_direction,
                                              distance=distance),
                          direction=new_direction)


class AgentDynamics(object):
    def __init__(self, forward_speed: float, turn_speed: float):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed

    def __iter__(self):
        yield self.forward_speed
        yield self.turn_speed

    def change(self, delta_t: float) -> tuple:
        return self.forward_speed * delta_t,  self.turn_speed * delta_t


class Agent(object):
    class RenderMode(object):
        SPRITE = 0
        POLYGON = 1

    def __init__(self,
                 view_field: float = 360,
                 collision: bool = True,
                 size: float = .05,
                 sprite_scale: float = 1.0,
                 polygon_color: typing.Tuple[int, int, int] = (0, 80, 120)):
        self.visible = True
        self.render_mode = Agent.RenderMode.SPRITE
        self.view_field = view_field
        self._state: AgentState = AgentState()
        self.dynamics: AgentDynamics = AgentDynamics(forward_speed=0,
                                                     turn_speed=0)
        self.polygon = self.create_polygon()
        self.polygon_color = polygon_color
        self.collision = collision

        self.size = size

        self.sprite = None
        self.sprite_scale = sprite_scale
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {"reset": [],
                                                                               "step": []}
        self.on_reset = None
        self.on_step = None
        self.on_start = None
        self.name = ""
        self.model = None
        self.running = False
        self.data = None

    def set_state(self, state: AgentState) -> None:
        self._state = state

    @property
    def state(self) -> AgentState:
        return self._state

    def __handle_event__(self, event_name, *args):
        for handler in self.event_handlers[event_name]:
            handler(*args)

    def add_event_handler(self, event_name, handler):
        if event_name not in self.event_handlers:
            raise KeyError(f'Event handler "{event_name}" not defined')
        self.event_handlers[event_name].append(handler)

    def reset(self) -> None:
        self.__handle_event__("reset")
        self.running = True

    def step(self, delta_t: float) -> None:
        self.__handle_event__("step", delta_t)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("agent.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 90)
        return rotated_sprite

    @staticmethod
    def create_polygon() -> Polygon:
        return Polygon.regular((0, 0), .05, 30, sides=6)

    def get_polygon(self,
                    state: AgentState = None) -> Polygon:
        # Rotate and then translate the arrow polygon
        if state:
            return self.polygon.translate_rotate(translation=state.location, rotation=state.direction)
        else:
            return self.polygon.translate_rotate(translation=self.state.location, rotation=self.state.direction)

    def render(self,
               surface: pygame.Surface,
               coordinate_converter: CoordinateConverter):
        if self.visible:
            if self.render_mode == Agent.RenderMode.SPRITE:
                if self.sprite is None:
                    sprite_size = coordinate_converter.scale_from_canonical(self.size) * self.sprite_scale
                    self.sprite = pygame.transform.scale(self.create_sprite(), (sprite_size, sprite_size))
                rotated_sprite = pygame.transform.rotate(self.sprite, self._state.direction)
                width, height = rotated_sprite.get_size()
                screen_x, screen_y = coordinate_converter.from_canonical(self.state.location)
                surface.blit(rotated_sprite, (screen_x - width / 2, screen_y - height / 2))
            else:
                self.get_polygon().render(surface=surface,
                                          coordinate_converter=coordinate_converter,
                                          color=self.polygon_color)
