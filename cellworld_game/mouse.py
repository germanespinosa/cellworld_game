import math
import typing

import pygame
from .agent import AgentState, Agent
from .navigation import Navigation
from .navigation_agent import NavigationAgent
from .resources import Resources
import shapely as sp
from .util import distance


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
        self.observation = None
        self.finished = False
        self.puffed = False

    def get_observation(self):
        observation = Agent.get_observation(self=self)
        return self.parse_observation(observation=observation)

    def parse_observation(self, observation: dict):
        goal_distance = distance(self.goal_location, self.state.location)
        self.finished = goal_distance <= self.goal_threshold
        parsed_observation = [self.state.location[0],
                              self.state.location[1],
                              math.radians(self.state.direction)]

        if observation["agent_states"]["predator"]:
            predator_distance = distance(self.state.location,
                                         self.predator.state.location)
            parsed_observation.append(self.predator.state.location[0])
            parsed_observation.append(self.predator.state.location[1])
            parsed_observation.append(math.radians(self.predator.state.direction))
            parsed_observation.append(goal_distance)
            parsed_observation.append(predator_distance)
        else:
            parsed_observation += [0,
                                   0,
                                   0,
                                   goal_distance,
                                   0]

        parsed_observation += [o[0] for o in observation["walls"][:3]]
        parsed_observation += [math.radians(o[1]) for o in observation["walls"][:3]]
        parsed_observation += [self.puffed, self.puff_cool_down, self.finished]
        return parsed_observation

    def reset(self):
        self.finished = False
        self.puff_cool_down = 0
        self.observation = None
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
                self.puffed = True
                self.puff_cool_down = self.puff_cool_down_time

        if delta_t < self.puff_cool_down:
            self.puff_cool_down -= delta_t
        else:
            self.puff_cool_down = 0
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

