import random
import typing
import pygame
from .navigation import Navigation
from .navigation_agent import NavigationAgent
from .resources import Resources
import shapely as sp
import time


class Robot(NavigationAgent):
    def __init__(self,
                 start_locations: typing.List[typing.Tuple[float, float]],
                 open_locations: typing.List[typing.Tuple[float, float]],
                 navigation: Navigation):
        NavigationAgent.__init__(self,
                                 navigation=navigation)
        self.start_locations = start_locations
        self.open_locations = open_locations
        self.last_destination_time = 0

    def reset(self):
        self.state.location = random.choice(self.start_locations)
        self.state.direction = 180

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("predator.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 270)
        return rotated_sprite

    @staticmethod
    def create_polygon() -> sp.Polygon:
        return sp.Polygon([(.02, 0.013), (-.02, 0.013), (-.02, -0.013), (.02, -0.013), (.025, -0.01), (.025, 0.01)])

    def step(self, delta_t: float):
        if self.last_destination_time + 1 < time.time():
            observation = self.get_observation()
            self.last_destination_time = time.time()
            if "prey" in observation["agent_states"] and observation["agent_states"]["prey"]:
                self.set_destination(observation["agent_states"]["prey"][0])

            if not self.path:
                self.set_destination(random.choice(self.open_locations))
        self.navigate(delta_t=delta_t)
