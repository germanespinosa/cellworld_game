import math
import typing

import pygame
from .agent import AgentState, Agent
from .navigation import Navigation
from .navigation_agent import NavigationAgent
from .resources import Resources
import shapely as sp
from enum import Enum
from .util import distance


class MouseObservation(typing.List[float]):
    class Field(Enum):
        prey_x = 0
        prey_y = 1
        prey_direction = 2
        predator_x = 3
        predator_y = 4
        predator_direction = 5
        goal_distance = 6
        predator_distance = 7
        puffed = 8
        puff_cooled_down = 9
        finished = 10

    def __init__(self):
        super().__init__()
        for i in MouseObservation.Field:
            self.append(0.0)

    def __setitem__(self, field: Field, value):
        list.__setitem__(self, field.value, value)


class Mouse(NavigationAgent):
    def __init__(self,
                 start_state: AgentState,
                 actions: typing.List[typing.Tuple[float, float]],
                 goal_location: typing.Tuple[float, float],
                 goal_threshold: float,
                 puff_threshold: float,
                 puff_cool_down_time: float,
                 navigation: Navigation,
                 predator: Agent):
        NavigationAgent.__init__(self,
                                 navigation=navigation,
                                 max_forward_speed=0.5,
                                 max_turning_speed=20.0)
        self.start_state = start_state
        self.actions = actions
        self.goal_location = goal_location
        self.goal_threshold = goal_threshold
        self.puff_threshold = puff_threshold
        self.puff_cool_down = 0
        self.puff_cool_down_time = puff_cool_down_time
        self.predator = predator
        self.finished = False
        self.puffed = False

    def reset(self):
        self.finished = False
        self.puff_cool_down = 0
        self.state.location = self.start_state.location
        self.state.direction = self.start_state.direction
        NavigationAgent.reset(self)

    def start(self):
        NavigationAgent.start(self)

    def step(self, delta_t: float):
        if self.puff_cool_down <= 0:
            predator_distance = distance(self.state.location,
                                         self.predator.state.location)
            if predator_distance <= self.puff_threshold:
                if self.model.visibility.line_of_sight(self.state.location, self.predator.state.location):
                    self.puffed = True
                    self.puff_cool_down = self.puff_cool_down_time

        if delta_t < self.puff_cool_down:
            self.puff_cool_down -= delta_t
        else:
            self.puff_cool_down = 0

        goal_distance = distance(self.goal_location, self.state.location)
        self.finished = goal_distance <= self.goal_threshold
        self.navigate(delta_t=delta_t)

    @staticmethod
    def create_sprite() -> pygame.Surface:
        sprite = pygame.image.load(Resources.file("prey.png"))
        rotated_sprite = pygame.transform.rotate(sprite, 270)
        return rotated_sprite

    @staticmethod
    def create_polygon() -> sp.Polygon:
        return sp.Polygon([(.015, 0), (0, 0.005), (-.015, 0), (0, -0.005)])

    def set_action(self, action_number: int):
        self.set_destination(self.actions[action_number])

