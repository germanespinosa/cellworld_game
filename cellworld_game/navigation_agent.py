import typing

import pygame

from .util import distance, direction, direction_difference, direction_error_normalization
from .agent import Agent
from .navigation import Navigation


class NavigationAgent(Agent):

    def __init__(self,
                 navigation: Navigation,
                 max_forward_speed: float,
                 max_turning_speed: float,
                 threshold: float = 0.01):
        self.max_forward_speed = max_forward_speed
        self.max_turning_speed = max_turning_speed
        self.threshold = threshold
        self.navigation = navigation
        self.destination = None
        self.path = []
        Agent.__init__(self)
        self.collision = False

    def next_step(self):
        if self.path:
            return self.path[0]
        return None

    def set_destination(self, destination):
        if destination != self.destination:
            self.destination = destination
            self.path = self.navigation.get_path(src=self.state.location, dst=self.destination)

    def reset(self):
        self.destination = None
        self.path = []
        Agent.reset(self)

    def start(self):
        Agent.start(self)

    def navigate(self, delta_t: float):

        if self.next_step() is not None:
            distance_error = distance(src=self.state.location,
                                      dst=self.next_step())
            if distance_error < self.threshold:
                self.path.pop(0)

        if self.next_step():
            distance_error = distance(src=self.state.location,
                                      dst=self.next_step())

            normalized_distance_error = max(distance_error/.2, 1)

            destination_direction = direction(src=self.state.location,
                                              dst=self.next_step())
            direction_error = direction_difference(direction1=self.state.direction,
                                                   direction2=destination_direction)
            normalized_direction_error = direction_error_normalization(direction_error=direction_error)

            self.dynamics.forward_speed = self.max_forward_speed * normalized_direction_error * normalized_distance_error
            self.dynamics.turn_speed = self.max_turning_speed * direction_error
        else:
            self.dynamics.forward_speed = 0
            self.dynamics.turn_speed = 0

    def render(self,
               surface: pygame.Surface):
        current_point = self.state.location
        for step in self.path:
            pygame.draw.line(surface,
                             (255, 0, 0),
                             self.coordinate_converter.from_canonical(current_point),
                             self.coordinate_converter.from_canonical(step),
                             2)
            pygame.draw.circle(surface=surface,
                               color=(0, 0, 255),
                               center=self.coordinate_converter.from_canonical(step),
                               radius=5,
                               width=2)
            current_point = step
        Agent.render(self=self, surface=surface)
