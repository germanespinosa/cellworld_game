from .model import Model
from .robot import Robot
from .mouse import Mouse, MouseObservation
from .agent import AgentState
from .view import View
from gymnasium import Env
from gymnasium import spaces
from .cellworld_loader import CellWorldLoader
import numpy as np
from .util import distance
import math


class Environment(Env):
    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 200,
                 reward_function=lambda x: 0,
                 step_wait: int = 5,
                 render: bool = False,
                 real_time: bool = False):
        self.max_step = max_step
        self.reward_function = reward_function
        self.step_wait = step_wait
        self.loader = CellWorldLoader(world_name=world_name)
        self.observation = MouseObservation()
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.loader.tlppo_action_list)
                                            if use_lppos
                                            else len(self.loader.open_locations))
        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        self.model = Model(arena=self.loader.arena,
                           occlusions=self.loader.occlusions,
                           time_step=.025,
                           real_time=real_time)
        if use_predator:
            self.predator = Robot(start_locations=self.loader.robot_start_locations,
                                  open_locations=self.loader.open_locations,
                                  navigation=self.loader.navigation)
            self.model.add_agent("predator", self.predator)

        self.prey = Mouse(start_state=AgentState(location=(.05, .5),
                                                 direction=0),
                          goal_location=(1, .5),
                          goal_threshold=.1,
                          puff_threshold=.1,
                          puff_cool_down_time=.5,
                          navigation=self.loader.navigation,
                          actions=self.action_list,
                          predator=self.predator)
        self.model.add_agent("prey", self.prey)
        self.view = None
        self.render_steps = render
        self.step_count = 0
        self.captures = 0
        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0

    def get_observation(self):
        self.observation[MouseObservation.Field.prey_x] = self.prey.state.location[0]
        self.observation[MouseObservation.Field.prey_y] = self.prey.state.location[1]
        self.observation[MouseObservation.Field.prey_direction] = math.radians(self.prey.state.direction)

        if self.model.visibility.line_of_sight(self.prey.state.location, self.predator.state.location):
            self.observation[MouseObservation.Field.predator_x] = self.predator.state.location[0]
            self.observation[MouseObservation.Field.predator_y] = self.predator.state.location[1]
            self.observation[MouseObservation.Field.predator_direction] = math.radians(
                self.predator.state.direction)
            predator_distance = distance(self.prey.state.location, self.predator.state.location)
        else:
            self.observation[MouseObservation.Field.predator_x] = 0
            self.observation[MouseObservation.Field.predator_y] = 0
            self.observation[MouseObservation.Field.predator_direction] = 0
            predator_distance = 1

        goal_distance = distance(self.prey.goal_location, self.prey.state.location)
        self.observation[MouseObservation.Field.goal_distance] = goal_distance
        self.observation[MouseObservation.Field.predator_distance] = predator_distance
        self.observation[MouseObservation.Field.puffed] = self.prey.puffed
        self.observation[MouseObservation.Field.puff_cooled_down] = self.prey.puff_cool_down
        self.observation[MouseObservation.Field.finished] = self.prey.finished
        return self.observation

    def set_action(self, action: int):
        self.prey.set_action(action)

    def step(self, action: int):
        self.step_count += 1
        self.set_action(action=action)
        for i in range(self.step_wait):
            self.model.step()
            if self.render_steps:
                self.render()
        truncated = (self.step_count >= self.max_step)
        obs = self.get_observation()
        reward = self.reward_function(obs)
        if self.prey.puffed:
            self.captures += 1
            self.prey.puffed = False
        if self.prey.finished or truncated:
            info = {"captures": self.captures,
                    "is_success": 1 if self.prey.finished and self.captures == 0 else 0,
                    "survived": 1 if self.prey.finished and self.captures == 0 else 0,
                    "agents": {}}
            for agent_name, agent in self.model.agents.items():
                info["agents"][agent_name] = {}
                info["agents"][agent_name] = agent.get_stats()
        else:
            info = {}
        return obs, reward, self.prey.finished, truncated, info

    def reset(self, seed=None):
        self.captures = 0
        self.step_count = 0
        self.model.reset()
        obs = self.get_observation()
        return obs, {}

    def render(self):
        if self.view is None:
            self.view = View(model=self.model)
        self.view.draw()
