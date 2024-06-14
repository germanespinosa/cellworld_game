import random
import typing
import pygame
from .agent import AgentState
from .navigation import Navigation
from .navigation_agent import NavigationAgent
from .resources import Resources
from .polygon import Polygon
from .util import Point


class Robot(NavigationAgent):
    def __init__(self,
                 start_locations: typing.List[Point.type],
                 open_locations: typing.List[Point.type],
                 navigation: Navigation,
                 view_field: float = 360):
        NavigationAgent.__init__(self,
                                 navigation=navigation,
                                 max_forward_speed=0.075,
                                 max_turning_speed=3.5,
                                 view_field=view_field,
                                 size=.05,
                                 sprite_scale=1.9,
                                 polygon_color=(90, 20, 20))
        self.start_locations = start_locations
        self.open_locations = open_locations
        self.last_destination_time = 0

    def reset(self):
        NavigationAgent.reset(self)
        return AgentState(location=random.choice(self.start_locations), direction=180)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("predator.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 270)
        return rotated_sprite

    @staticmethod
    def create_body_polygon() -> Polygon:
        return Polygon([(.02, 0.013), (-.02, 0.013), (-.02, -0.013), (.02, -0.013), (.025, -0.01), (.025, 0.01)])

    def step(self, delta_t: float):
        NavigationAgent.step(self=self,
                             delta_t=delta_t)
