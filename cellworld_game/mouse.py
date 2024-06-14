import pygame
from .agent import AgentState
from .navigation import Navigation
from .navigation_agent import NavigationAgent
from .resources import Resources
from .polygon import Polygon


class Mouse(NavigationAgent):
    def __init__(self,
                 start_state: AgentState,
                 navigation: Navigation,
                 view_field: float = 360):
        NavigationAgent.__init__(self,
                                 navigation=navigation,
                                 max_forward_speed=0.5,
                                 max_turning_speed=20.0,
                                 view_field=view_field,
                                 size=0.04,
                                 sprite_scale=2.0,
                                 polygon_color=(20, 90, 20))
        self.start_state = start_state

    def reset(self):
        NavigationAgent.reset(self)
        return AgentState(location=self.start_state.location, direction=self.start_state.direction)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("prey.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 270)
        return rotated_sprite

    @staticmethod
    def create_body_polygon() -> Polygon:
        return Polygon([(.015, 0), (0, 0.005), (-.015, 0), (0, -0.005)])

