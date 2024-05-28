import typing
import pygame
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

    def __init__(self,
                 view_field: float = 360,
                 collision: bool = True):
        self.visible = True
        self.view_field = view_field
        self._state: AgentState = AgentState()
        self.dynamics: AgentDynamics = AgentDynamics(forward_speed=0,
                                                     turn_speed=0)
        self.polygon = self.create_polygon()
        self.sprite = None
        self.collision = collision
        self.on_reset = None
        self.on_step = None
        self.on_start = None
        self.name = ""
        self.model = None
        self.trajectory: typing.List[AgentState] = []
        self.coordinate_converter: typing.Optional[CoordinateConverter] = None
        self.running = False

    def set_sprite_size(self, size: tuple):
        self.sprite = pygame.transform.scale(self.create_sprite(), size)

    def set_state(self, state: AgentState) -> None:
        self.trajectory.append(state)
        self._state = state

    @property
    def state(self) -> AgentState:
        return self._state

    def reset(self) -> None:
        self.trajectory.clear()
        if self.on_reset:
            self.on_reset()
        self.running = True

    def start(self) -> None:
        if self.on_start:
            self.on_start()

    def step(self, delta_t: float) -> None:
        if self.on_step:
            self.on_step(delta_t)

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

    def get_sprite(self) -> pygame.Surface:
        rotated_sprite = pygame.transform.rotate(self.sprite, self._state.direction)
        return rotated_sprite

    def get_observation(self) -> dict:
        if self.model:
            return self.model.get_observation(agent_name=self.name)
        else:
            return None

    def get_stats(self) -> dict:
        stats = {}
        dist = 0
        prev_state = self.trajectory[0]
        for state in self.trajectory[1:]:
            dist += Point.distance(src=prev_state.location, dst=state.location)
            prev_state = state
        stats["distance"] = dist
        return stats

    def render(self,
               surface: pygame.Surface,
               coordinate_converter: CoordinateConverter):
        if self.visible:
            agent_sprite: pygame.Surface = self.get_sprite()
            width, height = agent_sprite.get_size()
            screen_x, screen_y = coordinate_converter.from_canonical(self.state.location)
            surface.blit(agent_sprite, (screen_x - width / 2, screen_y - height / 2))

    def set_coordinate_converter(self,
                                 coordinate_converter: CoordinateConverter):
        self.coordinate_converter = coordinate_converter
