import random
from ..util import Point
from ..model import Model
from ..agent import AgentState, CoordinateConverter
from ..mouse import Mouse
from ..robot import Robot
from ..cellworld_loader import CellWorldLoader


class BotEvade(Model):
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

        self.register_event(event_name="puff")

        if use_predator:
            self.predator = Robot(start_locations=self.loader.robot_start_locations,
                                  open_locations=self.loader.open_locations,
                                  navigation=self.loader.navigation)

            self.add_agent("predator", self.predator)

        self.prey = Mouse(start_state=AgentState(location=(.05, .5),
                                                 direction=0),
                          navigation=self.loader.navigation)

        self.add_agent("prey", self.prey)

        self.running = False

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

        self.puffed: bool = False
        self.puff_cool_down: float = 0
        self.goal_achieved: bool = False
        self.predator_prey_distance: float = 1
        self.prey_goal_distance: float = 0
        self.puff_count = 0
        self.predator_visible = False

    def __update_state__(self,
                         delta_t: float = 0):
        if self.use_predator and self.puff_cool_down <= 0:
            self.predator_prey_distance = Point.distance(src=self.prey.state.location,
                                                         dst=self.predator.state.location)
            self.predator_visible = self.line_of_sight["prey"]["predator"]
            if self.predator_visible:
                if self.predator_prey_distance <= self.puff_threshold:
                    self.puffed = True
                    self.puff_count += 1
                    self.puff_cool_down = self.puff_cool_down_time
                    self.__dispatch__("puff", self)

                self.predator.set_destination(self.prey.state.location)

            if not self.predator.path:
                self.predator.set_destination(random.choice(self.loader.open_locations))

        if delta_t < self.puff_cool_down:
            self.puff_cool_down -= delta_t
        else:
            self.puff_cool_down = 0

        self.prey_goal_distance = Point.distance(src=self.goal_location,
                                                 dst=self.prey.state.location)

        if self.prey_goal_distance <= self.goal_threshold:
            self.goal_achieved = True
            self.stop()

    def __on_quit__(self):
        self.stop()

    def reset(self):
        Model.reset(self)
        self.goal_achieved = False
        self.predator_visible = False
        self.puff_count = 0
        self.__update_state__()

    def step(self) -> float:
        delta_t = Model.step(self)
        if self.render:
            self.view.render()
        self.__update_state__(delta_t=delta_t)
        return delta_t
