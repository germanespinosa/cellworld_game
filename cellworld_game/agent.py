import typing
import pygame
import shapely as sp
from .resources import Resources
from .util import create_hexagon, move_point
from shapely.affinity import rotate, translate


class AgentState(object):
    def __init__(self, location: typing.Tuple[float, float] = (0, 0), direction: float = 0):
        self.location = location
        self.direction = direction

    def __iter__(self):
        yield self.location
        yield self.direction

    def update(self,
               distance: float,
               rotation: float) -> "AgentState":
        new_direction = self.direction + rotation
        return AgentState(location=move_point(start=self.location,
                                              direction=new_direction,
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
                 view_field: float = 180,
                 collision: bool = True):
        self.view_field = view_field
        self.state: AgentState = AgentState()
        self.dynamics: AgentDynamics = AgentDynamics(forward_speed=0,
                                                     turn_speed=0)
        self.polygon = self.create_polygon()
        sprite = self.create_sprite()
        self.sprite = pygame.transform.scale(sprite, (75, 75))
        self.collision = collision
        self.on_reset = None
        self.on_step = None
        self.on_start = None
        self.name = ""
        self.model = None

    def reset(self):
        if self.on_reset:
            self.on_reset()

    def start(self):
        if self.on_start:
            self.on_start()

    def step(self, delta_t: float):
        if self.on_step:
            self.on_step(delta_t)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("agent.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 90)
        return rotated_sprite

    @staticmethod
    def create_polygon() -> sp.Polygon:
        return create_hexagon((0, 0), .05, 30)

    def get_polygon(self,
                    state: AgentState = None) -> sp.Polygon:
        # Rotate and then translate the arrow polygon
        if state:
            x, y = state.location
            direction = state.direction
        else:
            x, y = self.state.location
            direction = self.state.direction
        rotated_polygon = rotate(self.polygon,
                                 direction,
                                 origin=(0, 0),
                                 use_radians=False)
        translated_polygon = translate(rotated_polygon, x, y)
        return translated_polygon

    def get_sprite(self) -> pygame.Surface:
        rotated_sprite = pygame.transform.rotate(self.sprite, self.state.direction)
        return rotated_sprite

    def get_observation(self):
        if self.model:
            return self.model.get_observation(agent_name=self.name)
        else:
            return None