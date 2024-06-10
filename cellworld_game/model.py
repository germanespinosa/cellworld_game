import time
import typing
import shapely as sp
from .agent import Agent, AgentState
from .visibility import Visibility
from .polygon import Polygon


class Model(object):

    def __init__(self,
                 world_name: str,
                 arena: Polygon,
                 occlusions: typing.List[Polygon],
                 time_step: float = 0.1,
                 real_time: bool = False,
                 render: bool = False):
        self.world_name = world_name
        self.arena = arena
        self.occlusions = occlusions
        self.time_step = time_step
        self.real_time = real_time
        self.render = render
        self.agents: typing.Dict[str, Agent] = {}
        self.visibility = Visibility(arena=self.arena, occlusions=self.occlusions)
        self.last_step = None
        self.time = 0
        self.running = False
        self.episode_count = 0
        self.step_count = 0
        self.view: typing.Optional["View"] = None
        self.paused = False
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {"before_step": [], "after_step": [],
                                                                               "before_stop": [], "after_stop": [],
                                                                               "before_reset": [], "after_reset": [],
                                                                               "close": [], "update_agents_polygons": [],
                                                                               "pause": [], "update_agents_visibilities": []}
        self.agents_visibility: typing.Dict[str, typing.Dict[str, typing.Union[IPolygon, typing.Dict[str, bool]]]] = {}
        self.agents_polygons: typing.Dict[str, Polygon] = {}
        if self.render:
            self.occlusion_color = (50, 50, 50)
            self.arena_color = (210, 210, 210)
            self.visibility_color = (255, 255, 255)
            from .view import View
            self.view = View(model=self)
            self.view.add_event_handler("quit", self.close)

            def render_occlusions(surface, coordinate_converter):
                for occlusion in self.occlusions:
                    occlusion.render(surface=surface,
                                     coordinate_converter=coordinate_converter,
                                     color=self.occlusion_color)

            def render_arena(surface, coordinate_converter):
                self.arena.render(surface=surface,
                                  coordinate_converter=coordinate_converter,
                                  color=self.arena_color)

            self.view.add_render_step(render_step=render_arena, z_index=0)
            self.view.add_render_step(render_step=render_occlusions, z_index=30)

            self.agent_render_mode = Agent.RenderMode.SPRITE
            self.render_agent_visibility = ""

            def render_visibility(surface, coordinate_converter):
                if self.render_agent_visibility == "":
                    return
                visibility_polygon = self.agents_visibility[self.render_agent_visibility]["polygon"]
                visibility_polygon.render(surface=surface,
                                          coordinate_converter=coordinate_converter,
                                          color=self.visibility_color)

            def key_down(key):
                import pygame
                if key == pygame.K_r:
                    if self.agent_render_mode == Agent.RenderMode.SPRITE:
                        self.agent_render_mode = Agent.RenderMode.POLYGON
                    else:
                        self.agent_render_mode = Agent.RenderMode.SPRITE
                    for agent_name, agent in self.agents.items():
                        agent.render_mode = self.agent_render_mode
                elif key == pygame.K_v:
                    agent_names = list(self.agents.keys())
                    if self.render_agent_visibility == "":
                        self.render_agent_visibility = agent_names[0]
                    else:
                        current_agent = agent_names.index(self.render_agent_visibility)
                        if current_agent == len(agent_names) - 1:
                            self.render_agent_visibility = ""
                        else:
                            self.render_agent_visibility = agent_names[current_agent + 1]
                elif key == pygame.K_p:
                    self.pause()

            self.view.add_render_step(render_visibility, z_index=20)

            self.view.add_event_handler("key_down", key_down)

    def set_agents_state(self, agents_state: typing.Dict[str, AgentState],
                         delta_t: float = 0):
        for agent_name, agent_state in agents_state.items():
            if agent_name in self.agents:
                self.agents[agent_name].set_state(agent_state)
        self.__update_state__(delta_t)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise "Event handler not registered"
        self.event_handlers[event_name].append(handler)

    def __handle_event__(self, event_name: str, *args):
        for event_handler in self.event_handlers[event_name]:
            event_handler(*args)

    def __update_state__(self,
                         delta_t: float = 0):
        pass

    def __update_agents_visibilities__(self):
        for agent_name, agent in self.agents.items():
            agent_visibility_polygon = self.visibility.get_visibility_polygon(src=agent.state.location,
                                                                              direction=agent.state.direction,
                                                                              view_field=agent.view_field)
            self.agents_visibility[agent_name]["polygon"] = agent_visibility_polygon
            for other_agent_name, other_agent_polygon in self.agents_polygons.items():
                if agent_name != other_agent_name:
                    self.agents_visibility[agent_name]["agents"][other_agent_name] = agent_visibility_polygon.contains(other_agent_polygon)
        self.__handle_event__("update_agents_visibilities", self.agents_visibility)

    def __update_agents_polygons__(self):
        for agent_name, agent in self.agents.items():
            self.agents_polygons[agent_name] = agent.get_polygon()
        self.__handle_event__("update_agents_polygons", self.agents_polygons)

    def pause(self):
        self.paused = not self.paused
        self.__handle_event__("pause", self)

    def add_agent(self, name: str, agent: Agent):
        self.agents[name] = agent
        agent.name = name
        agent.model = self
        self.agents_polygons[name] = None
        self.agents_visibility[name] = {"polygon": None, "agents": {}}
        if self.render:
            self.view.add_render_step(agent.render, z_index=100)

    def reset(self):
        if self.running:
            self.stop()
        self.__handle_event__("before_reset", self)
        self.running = True
        self.episode_count += 1
        for name, agent in self.agents.items():
            agent.reset()
        self.last_step = time.time()
        self.step_count = 0
        self.__update_agents_polygons__()
        self.__update_agents_visibilities__()
        self.__handle_event__("after_reset", self)

    def stop(self):
        if not self.running:
            return
        self.__handle_event__("before_stop", self)
        self.running = False
        self.__handle_event__("after_stop", self)

    def is_valid_state(self, agent_polygon: sp.Polygon, collisions: bool) -> bool:
        if not self.arena.contains(agent_polygon):
            return False
        if collisions:
            for occlusion in self.occlusions:
                if agent_polygon.intersects(occlusion):
                    return False
        return True

    def step(self) -> float:
        if not self.running:
            return 0

        if self.paused:
            return 0

        self.__handle_event__("before_step", self)

        if self.real_time:
            while self.last_step + self.time_step > time.time():
                pass

        self.last_step = time.time()
        for name, agent in self.agents.items():
            dynamics = agent.dynamics
            distance, rotation = dynamics.change(delta_t=self.time_step)
            new_state = agent.state.update(rotation=rotation,
                                           distance=distance)
            agent_polygon = agent.get_polygon(state=new_state)
            if self.is_valid_state(agent_polygon=agent_polygon,
                                   collisions=agent.collision):
                agent.set_state(state=new_state)
            else: #try only rotation
                new_state = agent.state.update(rotation=rotation,
                                               distance=0)
                agent_polygon = agent.get_polygon(state=new_state)
                if self.is_valid_state(agent_polygon=agent_polygon,
                                       collisions=agent.collision):
                    agent.set_state(state=new_state)
                else: #try only translation
                    new_state = agent.state.update(rotation=0,
                                                   distance=distance)
                    agent_polygon = agent.get_polygon(state=new_state)
                    if self.is_valid_state(agent_polygon=agent_polygon,
                                           collisions=agent.collision):
                        agent.set_state(state=new_state)
        self.__update_agents_polygons__()
        self.__update_agents_visibilities__()
        for name, agent in self.agents.items():
            agent.step(delta_t=self.time_step)
        self.time += self.time_step
        self.step_count += 1
        self.__handle_event__("after_step", self)
        return self.time_step

    def close(self):
        if self.running:
            self.stop()
        self.__handle_event__("close", self)
