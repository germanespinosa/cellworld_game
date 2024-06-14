import time
import typing
import shapely as sp
from .agent import Agent, AgentState
from .visibility import Visibility
from .polygon import Polygon
from .event import EventDispatcher
from .line_of_sight import LineOfSight


class Model(EventDispatcher):

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
        self.time: float = 0
        self.running = False
        self.episode_count = 0
        self.step_count = 0
        self.view: typing.Optional["View"] = None
        self.paused = False
        self.line_of_sight = LineOfSight()
        EventDispatcher.__init__(self, ["before_step",
                                        "after_step",
                                        "before_stop",
                                        "after_stop",
                                        "before_reset",
                                        "after_reset",
                                        "agents_states_update",
                                        "close",
                                        "pause"])
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
                visibility_polygon = self.agents[self.render_agent_visibility].visibility_polygon
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
                elif key == pygame.K_n:
                    for agent_name, agent in self.agents.items():
                        if hasattr(agent, "render_path"):
                            agent.render_path = not agent.render_path

            self.view.add_render_step(render_visibility, z_index=20)

            self.view.add_event_handler("key_down", key_down)

    def set_agents_state(self,
                         agents_state: typing.Dict[str, AgentState],
                         agents_body_polygons: typing.Dict[str, Polygon] = None,
                         delta_t: float = 0):
        for agent_name, agent_state in agents_state.items():
            agent = self.agents[agent_name]
            agent.state = agent_state
            if agents_body_polygons:
                agent.body_polygon = agents_body_polygons[agent_name]
            else:
                agent.body_polygon = agent.get_body_polygon(state=agent_state)

            agent.visibility_polygon = self.visibility.get_visibility_polygon(src=agent_state.location,
                                                                              direction=agent_state.direction,
                                                                              view_field=agent.view_field)

            self.__dispatch__(f"agent_{agent_name}_state_update", agent_state)

        for agent_name, agent in self.agents.items():
            agent_visibility_polygon = self.agents[agent_name].visibility_polygon
            for other_agent_name, other_agent in self.agents.items():
                if other_agent_name != agent_name:
                    self.line_of_sight[agent_name, other_agent_name] = agent_visibility_polygon.intersects(other_agent.body_polygon)
        self.__dispatch__("agents_states_update", agents_state, self.line_of_sight)

    def pause(self):
        self.paused = not self.paused
        self.__dispatch__("pause", self)

    def add_agent(self, name: str, agent: Agent):
        agent.name = name
        agent.model = self
        self.register_event(f"agent_{name}_state_update")
        self.line_of_sight.register_agent(agent=agent)
        self.agents[name] = agent
        if self.render:
            self.view.add_render_step(agent.render, z_index=100)

    def reset(self):
        if self.running:
            self.stop()
        self.__dispatch__("before_reset", self)
        self.running = True
        self.episode_count += 1
        agents_state: typing.Dict[str, AgentState] = {}
        agents_body_polygon: typing.Dict[str, Polygon] = {}
        for name, agent in self.agents.items():
            agent_state = agent.reset()
            agents_state[name] = agent_state
            agents_body_polygon[name] = agent.get_body_polygon(state=agent_state)

        self.set_agents_state(agents_state=agents_state)
        self.last_step = time.time()
        self.step_count = 0
        self.__dispatch__("after_reset", self)

    def stop(self):
        if not self.running:
            return
        self.__dispatch__("before_stop", self)
        self.running = False
        self.__dispatch__("after_stop", self)

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

        self.__dispatch__("before_step", self)

        if self.real_time:
            while self.last_step + self.time_step > time.time():
                pass

        self.last_step = time.time()
        new_states: typing.Dict[str, AgentState] = {}
        new_body_polygons: typing.Dict[str, Polygon] = {}
        for name, agent in self.agents.items():
            dynamics = agent.dynamics
            distance, rotation = dynamics.change(delta_t=self.time_step)
            new_state = agent.state.update(rotation=rotation,
                                           distance=distance)
            agent_polygon = agent.get_body_polygon(state=new_state)
            if not self.is_valid_state(agent_polygon=agent_polygon,
                                       collisions=agent.collision): #try only rotation
                new_state = agent.state.update(rotation=rotation,
                                               distance=0)
                agent_polygon = agent.get_body_polygon(state=new_state)
                if not self.is_valid_state(agent_polygon=agent_polygon,
                                           collisions=agent.collision): #try only translation
                    new_state = agent.state.update(rotation=0,
                                                   distance=distance)
                    agent_polygon = agent.get_body_polygon(state=new_state)
                    if not self.is_valid_state(agent_polygon=agent_polygon,
                                               collisions=agent.collision):
                        new_state = agent.state
                        agent_polygon = agent.body_polygon
            new_states[name] = new_state
            new_body_polygons[name] = agent_polygon
        self.set_agents_state(agents_state=new_states, agents_body_polygons=new_body_polygons)
        for name, agent in self.agents.items():
            agent.step(delta_t=self.time_step)
        self.time += self.time_step
        self.step_count += 1
        self.__dispatch__("after_step", self)
        return self.time_step

    def close(self):
        if self.running:
            self.stop()
        self.__dispatch__("close", self)

    def __del__(self):
        self.close()
