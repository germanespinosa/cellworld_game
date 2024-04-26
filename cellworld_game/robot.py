import shapely as sp
import random
import typing
import pygame
from .agent import AgentState
from .navigation import Navigation
from .navigation_agent import NavigationAgent
from .resources import Resources


class Robot(NavigationAgent):
    def __init__(self,
                 start_locations: typing.List[typing.Tuple[float, float]],
                 open_locations: typing.List[typing.Tuple[float, float]],
                 navigation: Navigation):
        NavigationAgent.__init__(self,
                                 navigation=navigation,
                                 max_forward_speed=0.075,
                                 max_turning_speed=3.5)
        self.start_locations = start_locations
        self.open_locations = open_locations
        self.last_destination_time = 0

    def reset(self):
        NavigationAgent.reset(self)
        self.set_state(AgentState(location=random.choice(self.start_locations),
                                  direction=180))

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("predator.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 270)
        return rotated_sprite

    @staticmethod
    def create_polygon() -> sp.Polygon:
        return sp.Polygon([(.02, 0.013), (-.02, 0.013), (-.02, -0.013), (.02, -0.013), (.025, -0.01), (.025, 0.01)])

    def step(self, delta_t: float):
        NavigationAgent.step(self=self,
                             delta_t=delta_t)
