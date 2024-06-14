import pulsekit
import pygame
from .util import Point, Direction
from .agent import Agent
from .coordinate_converter import CoordinateConverter
from .navigation import Navigation
import typing


class NavigationAgent(Agent):

    def __init__(self,
                 navigation: Navigation,
                 max_forward_speed: float,
                 max_turning_speed: float,
                 threshold: float = 0.01,
                 destination_update_rate: int = 10,
                 view_field: float = 360,
                 size: float = 0.5,
                 sprite_scale: float = 1,
                 polygon_color: typing.Tuple[int, int, int] = (0, 80, 120)):
        self.max_forward_speed = max_forward_speed
        self.max_turning_speed = max_turning_speed
        self.threshold = threshold
        self.destination_update_rate = destination_update_rate
        self.navigation = navigation
        self.new_destination = None
        self.destination = None
        self.navigation_plan_update_wait = 0
        self.destination_wait = 0
        self.path = []
        self.render_path = False
        Agent.__init__(self,
                       view_field=view_field,
                       size=size,
                       sprite_scale=sprite_scale,
                       polygon_color=polygon_color)
        self.collision = False

    def next_step(self):
        if self.path:
            return self.path[0]
        return None

    def set_destination(self, destination):
        self.new_destination = destination

    def reset(self):
        self.destination = None
        self.path = []
        Agent.reset(self)

    def start(self):
        Agent.start(self)

    def step(self, delta_t: float):
        with pulsekit.CodeBlock("navigation_agent.step"):
            if not self.running:
                self.stop_navigation()
                return

            if self.new_destination and self.destination_wait == 0:
                self.destination = self.new_destination
                self.new_destination = None
                self.destination_wait = self.destination_update_rate
                self.navigation_plan_update_wait = self.destination_update_rate
                self.path = self.navigation.get_path(src=self.state.location, dst=self.destination)

            if self.destination and self.navigation_plan_update_wait == 0:
                self.path = self.navigation.get_path(src=self.state.location, dst=self.destination)

            if self.destination_wait:
                self.destination_wait -= 1

            if self.navigation_plan_update_wait:
                self.navigation_plan_update_wait -= 1

            if self.next_step() is not None:
                distance_error = Point.distance(src=self.state.location,
                                                dst=self.next_step())
                if distance_error < self.threshold:
                    self.path.pop(0)

            if self.next_step():
                distance_error = Point.distance(src=self.state.location,
                                                dst=self.next_step())

                normalized_distance_error = max(distance_error/.2, 1)

                destination_direction = Direction.degrees(src=self.state.location,
                                                          dst=self.next_step())
                direction_error = Direction.difference(direction1=self.state.direction,
                                                       direction2=destination_direction)
                normalized_direction_error = Direction.error_normalization(direction_error=direction_error)

                self.dynamics.forward_speed = self.max_forward_speed * normalized_direction_error * normalized_distance_error
                self.dynamics.turn_speed = self.max_turning_speed * direction_error
            else:
                self.dynamics.forward_speed = 0
                self.dynamics.turn_speed = 0

    def render(self,
               surface: pygame.Surface,
               coordinate_converter: CoordinateConverter):
        if self.render_path:
            current_point = self.state.location
            for step in self.path:
                if step is None:
                    continue
                pygame.draw.line(surface,
                                 (255, 0, 0),
                                 coordinate_converter.from_canonical(current_point),
                                 coordinate_converter.from_canonical(step),
                                 2)
                pygame.draw.circle(surface=surface,
                                   color=(0, 0, 255),
                                   center=coordinate_converter.from_canonical(step),
                                   radius=5,
                                   width=2)
                current_point = step
        Agent.render(self=self,
                     surface=surface,
                     coordinate_converter=coordinate_converter)

    def stop_navigation(self):
        self.dynamics.forward_speed = 0
        self.dynamics.turn_speed = 0
