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
                 goal_location_generator: typing.Callable[[bool], typing.Optional[typing.Tuple[float, float]]],
                 goal_threshold: float,
                 puff_threshold: float,
                 puff_cool_down_time: float,
                 navigation: Navigation,
                 predator: typing.Optional[Agent] = None):
        NavigationAgent.__init__(self,
                                 navigation=navigation,
                                 max_forward_speed=0.5,
                                 max_turning_speed=20.0)
        self.start_state = start_state
        self.actions = actions
        self.goal_location_generator = goal_location_generator
        self.goal_threshold = goal_threshold
        self.puff_threshold = puff_threshold
        self.puff_cool_down = 0
        self.puff_cool_down_time = puff_cool_down_time
        self.predator = predator
        self.finished = False
        self.goal_achieved = False
        self.puffed = False
        self.goal_location = None

    def reset(self):
        self.finished = False
        self.puff_cool_down = 0
        self.goal_location = self.goal_location_generator(True)
        NavigationAgent.reset(self)
        self.set_state(AgentState(location=self.start_state.location,
                                  direction=self.start_state.direction))

    def start(self):
        NavigationAgent.start(self)

    def step(self, delta_t: float):
        if self.finished:
            self.stop_navigation()
            return

        if self.puff_cool_down <= 0 and self.predator:
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
        if goal_distance <= self.goal_threshold:
            self.goal_achieved = True
            self.goal_location = self.goal_location_generator(False)

        if self.goal_location:
            self.navigate(delta_t=delta_t)
        else:
            self.finished = True

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

    def render(self,
               surface: pygame.Surface):
        if self.predator:
            predator_location = self.coordinate_converter.from_canonical(self.model.agents["predator"].state.location)
            puff_area_size = self.model.agents["prey"].puff_threshold * self.coordinate_converter.screen_width
            puff_location = predator_location[0] - puff_area_size, predator_location[1] - puff_area_size
            puff_area_surface = pygame.Surface((puff_area_size * 2, puff_area_size * 2), pygame.SRCALPHA)
            puff_area_color = (255, 0, 0, 60) if self.puff_cool_down > 0 else (0, 0, 255, 60)
            pygame.draw.circle(puff_area_surface,
                               color=puff_area_color,
                               center=(puff_area_size, puff_area_size),
                               radius=puff_area_size)
            surface.blit(puff_area_surface,
                             puff_location)
            pygame.draw.circle(surface=surface,
                               color=(0, 0, 255),
                               center=predator_location,
                               radius=puff_area_size,
                               width=2)
        NavigationAgent.render(self,
                               surface=surface)
