import random
from ..util import Point
from ..model import Model
from ..agent import AgentState, CoordinateConverter
from ..mouse import Mouse
from ..robot import Robot
from ..cellworld_loader import CellWorldLoader


class DualEvadePreyData:
    def __init__(self):
        self.puffed = False
        self.goal_achieved = False
        self.predator_visible = False
        self.predator_prey_distance = 1
        self.prey_goal_distance = 0
        self.puff_count = 0

    def reset(self):
        self.puffed = False
        self.goal_achieved = False
        self.predator_visible = False
        self.predator_prey_distance = 1
        self.prey_goal_distance = 0
        self.puff_count = 0


class DualEvade(Model):
    def __init__(self,
                 world_name: str = "21_05",
                 use_predator: bool = True,
                 puff_cool_down_time: float = .5,
                 puff_threshold: float = .1,
                 goal_location=(1.0, 0.5),
                 goal_threshold: float = .1,
                 time_step: float = .025,
                 real_time: bool = False,
                 render: bool = False):
        self.use_predator = use_predator
        self.puff_cool_down_time = puff_cool_down_time
        self.puff_threshold = puff_threshold
        self.goal_location = goal_location
        self.goal_threshold = goal_threshold
        self.loader = CellWorldLoader(world_name=world_name)

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

        self.prey_1 = Mouse(start_state=AgentState(location=(.05, .5),
                                                   direction=0),
                            navigation=self.loader.navigation)

        self.add_agent("prey_1", self.prey_1)

        self.prey_2 = Mouse(start_state=AgentState(location=(.05, .5),
                                                   direction=0),
                            navigation=self.loader.navigation)

        self.add_agent("prey_2", self.prey_2)

        if self.render:
            if use_predator:
                import pygame

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

        self.puff_cool_down: float = 0
        self.prey_data_1 = DualEvadePreyData()
        self.prey_data_2 = DualEvadePreyData()

    def __update_state__(self,
                         delta_t: float = 0):
        self.mouse_visible = self.visibility.line_of_sight(self.prey_1.state.location, self.prey_2.state.location)
        if self.use_predator and self.puff_cool_down <= 0:
            self.prey_data_1.predator_prey_distance = Point.distance(self.prey_1.state.location,
                                                                     self.predator.state.location)
            self.prey_data_1.predator_visible = self.prey_1.visible and self.visibility.line_of_sight(self.prey_1.state.location, self.predator.state.location)
            if self.prey_data_1.predator_visible:
                if self.prey_data_1.predator_prey_distance <= self.puff_threshold:
                    self.prey_data_1.puffed = True
                    self.prey_data_1.puff_count += 1
                    self.puff_cool_down = self.puff_cool_down_time

            self.prey_data_2.predator_prey_distance = Point.distance(self.prey_2.state.location,
                                                                     self.predator.state.location)

            self.prey_data_2.predator_visible = self.prey_2.visible and self.visibility.line_of_sight(self.prey_2.state.location, self.predator.state.location)
            if self.prey_data_2.predator_visible:
                if self.prey_data_2.predator_prey_distance <= self.puff_threshold:
                    self.prey_data_2.puffed = True
                    self.prey_data_2.puff_count += 1
                    self.puff_cool_down = self.puff_cool_down_time

            closest_visible_prey_location = None
            if self.prey_data_1.predator_visible:
                closest_visible_prey_location = self.prey_1.state.location

            if self.prey_data_2.predator_visible:
                if closest_visible_prey_location:
                    if self.prey_data_2.predator_prey_distance <= self.prey_data_1.predator_prey_distance:
                        closest_visible_prey_location = self.prey_2.state.location
                else:
                    closest_visible_prey_location = self.prey_2.state.location

            if closest_visible_prey_location:
                self.predator.set_destination(closest_visible_prey_location)

            if not self.predator.path:
                self.predator.set_destination(random.choice(self.loader.open_locations))

        if delta_t < self.puff_cool_down:
            self.puff_cool_down -= delta_t
        else:
            self.puff_cool_down = 0

        self.prey_data_1.prey_goal_distance = Point.distance(src=self.goal_location,
                                                             dst=self.prey_1.state.location)
        if self.prey_data_1.prey_goal_distance <= self.goal_threshold:
            self.prey_data_1.goal_achieved = True
            self.prey_1.visible = False

        self.prey_data_2.prey_goal_distance = Point.distance(src=self.goal_location,
                                                             dst=self.prey_2.state.location)
        if self.prey_data_2.prey_goal_distance <= self.goal_threshold:
            self.prey_data_2.goal_achieved = True
            self.prey_2.visible = False

        if self.prey_data_1.goal_achieved and self.prey_data_2.goal_achieved:
            self.stop()

    def __on_quit__(self):
        self.stop()

    def reset(self):
        self.prey_data_1.reset()
        self.prey_data_2.reset()
        self.prey_1.visible = True
        self.prey_2.visible = True
        Model.reset(self)
        self.__update_state__()

    def step(self) -> float:
        delta_t = Model.step(self)
        if self.render:
            self.view.render()
        self.__update_state__(delta_t=delta_t)
        return delta_t
