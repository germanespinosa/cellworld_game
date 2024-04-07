import math
import time
import typing
import shapely as sp
from .agent import Agent
from .visibility import Visibility
from .geometry import atan2


class Model(object):

    def __init__(self,
                 arena,
                 occlusions,
                 real_time: bool = False,
                 time_step: float = 0.1):
        self.arena = arena
        self.occlusions = occlusions
        self.real_time = real_time
        self.time_step = time_step
        self.agents: typing.Dict[str, Agent] = {}
        self.visibility = Visibility(arena=self.arena, occlusions=self.occlusions)
        self.last_step = None

    def add_agent(self, name: str, agent: Agent):
        self.agents[name] = agent
        agent.name = name
        agent.model = self

    def reset(self):
        for name, agent in self.agents.items():
            agent.reset()
        observations = self.get_observations()
        for name, agent in self.agents.items():
            agent.start()
        self.last_step = time.time()

    def is_valid_state(self, agent_polygon: sp.Polygon, collisions: bool) -> bool:
        if not self.arena.contains(agent_polygon):
            return False
        if collisions:
            for occlusion in self.occlusions:
                if agent_polygon.intersects(occlusion):
                    return False
        return True

    def wall_direction(self, src: typing.Tuple[float, float], wall_number: int):
        return math.degrees(atan2(sp.Point(src), self.visibility.walls_centroids[wall_number]))

    def get_observations(self):
        agent_visibility = {src_name: {dst_name: None for dst_name in self.agents} for src_name in self.agents}
        observations = {}
        for src_name, src_agent in self.agents.items():
            observations[src_name] = {}
            src_point = sp.Point(src_agent.state.location)
            walls_by_distance = self.visibility.walls_by_distance(src=src_point)
            parsed_walls = []
            for wall_number, vertices, distance in walls_by_distance:
                parsed_walls.append((distance, self.wall_direction(src=src_point, wall_number=wall_number)))
            observations[src_name]["walls"] = parsed_walls
            for dst_name, dst_agent in self.agents.items():
                if agent_visibility[src_name][dst_name] is None:
                    if src_name == dst_name:
                        is_visible = True
                    else:
                        dst_point = sp.Point(dst_agent.state.location)
                        is_visible = self.visibility.line_of_side(src=src_point,
                                                                  dst=dst_point,
                                                                  walls_by_distance=walls_by_distance)
                    agent_visibility[src_name][dst_name] = is_visible
                    agent_visibility[dst_name][src_name] = is_visible
            observations[src_name]["agent_states"] = {}
            for dst_name, is_visible in agent_visibility.items():
                if is_visible:
                    observations[src_name]["agent_states"][dst_name] = self.agents[dst_name].state.location, self.agents[dst_name].state.direction
                else:
                    observations[src_name]["agent_states"][dst_name] = None
        return observations

    def get_observation(self,
                        agent_name: str,
                        polygonal: bool = False) -> dict:
        observation = {}
        src_point = sp.Point(self.agents[agent_name].state.location)
        if polygonal:
            visibility_polygon = self.visibility.get_visibility_polygon(src_point.state.location, src_point.state.direction)
        else:
            walls_by_distance = self.visibility.walls_by_distance(src=src_point)
        parsed_walls = []
        for wall_number, vertices, distance in walls_by_distance:
            parsed_walls.append((distance, self.wall_direction(src=src_point, wall_number=wall_number)))
        observation["walls"] = parsed_walls
        observation["agent_states"] = {}
        for dst_name, dst_agent in self.agents.items():
            if agent_name == dst_name:
                is_visible = True
            else:
                dst_point = sp.Point(dst_agent.state.location)
                if polygonal:
                    dst_polygon = dst_agent.get_polygon()
                    is_visible = dst_polygon.intersects(visibility_polygon)
                else:
                    is_visible = self.visibility.line_of_side(src=src_point,
                                                              dst=dst_point,
                                                              walls_by_distance=walls_by_distance)
            if is_visible:
                observation["agent_states"][dst_name] = self.agents[dst_name].state.location, self.agents[dst_name].state.direction
            else:
                observation["agent_states"][dst_name] = None
        return observation

    def step(self):
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
                agent.state = new_state
            else: #try only rotation
                new_state = agent.state.update(rotation=rotation,
                                               distance=0)
                agent_polygon = agent.get_polygon(state=new_state)
                if self.is_valid_state(agent_polygon=agent_polygon,
                                       collisions=agent.collision):
                    agent.state = new_state
                else: #try only translation
                    new_state = agent.state.update(rotation=0,
                                                   distance=distance)
                    agent_polygon = agent.get_polygon(state=new_state)
                    if self.is_valid_state(agent_polygon=agent_polygon,
                                           collisions=agent.collision):
                        agent.state = new_state
        for name, agent in self.agents.items():
            agent.step(delta_t=self.time_step)
