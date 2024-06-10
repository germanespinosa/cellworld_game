import random
import typing

from ..util import Point
from ..model import Model
from ..agent import AgentState, CoordinateConverter
from ..mouse import Mouse
from ..robot import Robot
from ..cellworld_loader import CellWorldLoader


class Oasis(Model):
    def __init__(self,
                 world_name: str,
                 goal_locations: typing.List[typing.Tuple[float, float]],
                 goal_sequence_generator: typing.Callable[[None], typing.List[int]] = None,
                 use_predator: bool = True,
                 puff_cool_down_time: float = .5,
                 goal_time: float = 1,
                 puff_threshold: float = .1,
                 goal_threshold: float = .1,
                 time_step: float = .025,
                 real_time: bool = False,
                 render: bool = False):

        if goal_sequence_generator is None:
            goal_sequence_generator = lambda: random.sample(range(0, len(goal_locations)), 3)

        self.start_location = (.05, .5)
        self.goal_locations = goal_locations
        self.goal_time = goal_time
        self.goal_sequence_generator = goal_sequence_generator
        self.use_predator = use_predator
        self.puff_cool_down_time = puff_cool_down_time
        self.puff_threshold = puff_threshold
        self.goal_threshold = goal_threshold
        self.loader = CellWorldLoader(world_name=world_name)
        self.goal_location = None
        self.goal_sequence: typing.List[int] = []

        Model.__init__(self,
                       world_name=world_name,
                       arena=self.loader.arena,
                       occlusions=self.loader.occlusions,
                       time_step=time_step,
                       real_time=real_time,
                       render=render)

        if use_predator:
            self.predator = Robot(start_locations=self.loader.robot_start_locations,
                                  open_locations=self.loader.open_locations,
                                  navigation=self.loader.navigation)

            self.add_agent("predator", self.predator)

        self.prey = Mouse(start_state=AgentState(location=self.start_location,
                                                 direction=0),
                          navigation=self.loader.navigation)

        self.add_agent("prey", self.prey)

        if self.render:
            import pygame

            def render_active_goal(surface: pygame.Surface,
                                   coordinate_converter: CoordinateConverter):
                if self.goal_location is None:
                    return
                for goal_location in self.goal_locations:
                    screen_goal_location = coordinate_converter.from_canonical(goal_location)
                    radius = coordinate_converter.from_canonical(.0125)
                    goal_color = (0, 255, 0) if self.goal_location == goal_location else (255, 0, 0)
                    pygame.draw.circle(surface,
                                       color=goal_color,
                                       center=screen_goal_location,
                                       radius=radius)

            self.view.add_render_step(render_active_goal)

            if use_predator:
                def render_puff_area(surface: pygame.Surface,
                                     coordinate_converter: CoordinateConverter):
                    predator_location = coordinate_converter.from_canonical(self.predator.state.location)
                    puff_area_size = self.puff_threshold * coordinate_converter.screen_width
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

                self.view.add_render_step(render_puff_area, z_index=90)

        self.puffed: bool = False
        self.puff_cool_down: float = 0
        self.goal_achieved: bool = True
        self.goal_achieved_time: float = 0
        self.predator_prey_distance: float = 1
        self.prey_goal_distance: float = 0
        self.puff_count = 0

    def __update_state__(self,
                         delta_t: float = 0):
        if self.use_predator and self.puff_cool_down <= 0:
            self.predator_prey_distance = Point.distance(self.prey.state.location,
                                                         self.predator.state.location)
            if self.visibility.line_of_sight(self.prey.state.location, self.predator.state.location):
                if self.predator_prey_distance <= self.puff_threshold:
                    self.puffed = True
                    self.puff_count += 1
                    self.puff_cool_down = self.puff_cool_down_time

                self.predator.set_destination(self.prey.state.location)

            if not self.predator.path:
                self.predator.set_destination(random.choice(self.loader.open_locations))

        if delta_t < self.puff_cool_down:
            self.puff_cool_down -= delta_t
        else:
            self.puff_cool_down = 0

        self.prey_goal_distance = Point.distance(self.goal_location, self.prey.state.location)

        if self.prey_goal_distance <= self.goal_threshold:
            self.goal_achieved_time += delta_t
            if self.goal_achieved_time >= self.goal_time:
                self.__update_goal__()
                self.goal_achieved_time = 0
                self.goal_achieved = False
                if self.goal_location is None:
                    self.stop()
            else:
                self.goal_achieved = True

    def __on_quit__(self):
        self.stop()

    def __update_goal__(self):
        if self.goal_sequence:
            self.goal_location = self.goal_locations[self.goal_sequence.pop(0)]
        else:
            if self.goal_location == self.start_location:
                self.goal_location = None
            else:
                self.goal_location = self.start_location

    def reset(self):
        Model.reset(self)
        self.goal_achieved = False
        self.puff_count = 0
        self.goal_sequence = self.goal_sequence_generator()
        self.__update_goal__()
        self.__update_state__()

    def step(self) -> float:
        delta_t = Model.step(self)
        if self.render:
            self.view.render()
        self.__update_state__(delta_t=delta_t)
        return delta_t