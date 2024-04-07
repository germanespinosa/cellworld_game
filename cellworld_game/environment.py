from .model import Model
from .navigation import Navigation
from .robot import Robot
from .mouse import Mouse
from .agent import AgentState
from .view import View
from gymnasium import Env
from gymnasium import spaces
from .cellworld_loader import CellWorldLoader
import numpy as np


class Environment(Env):
    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 200,
                 reward_function=lambda x: 0,
                 step_wait: int = 5):
        self.max_step = max_step
        self.reward_function = reward_function
        self.step_wait = step_wait
        self.loader = CellWorldLoader(world_name=world_name)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.loader.tlppo_action_list)
                                            if use_lppos
                                            else len(self.loader.open_locations))

        self.model = Model(arena=self.loader.arena,
                           occlusions=self.loader.occlusions,
                           time_step=.1,
                           real_time=False)
        if use_predator:
            self.predator = Robot(start_locations=self.loader.robot_start_locations,
                                  open_locations=self.loader.open_locations,
                                  navigation=self.loader.navigation)
            self.model.add_agent("predator", self.predator)

        self.prey = Mouse(start_state=AgentState(location=(.05, .5),
                                                 direction=0),
                          goal_location=(1, .5),
                          goal_threshold=.1,
                          puff_threshold=.05,
                          puff_cool_down_time=.5,
                          navigation=self.loader.navigation,
                          actions=self.loader.full_action_list)
        self.model.add_agent("prey", self.prey)
        self.view = None

    def get_observation(self):
        return self.mouse.get_observation("prey")

    def set_action(self, action: int):
        self.prey.set_action(action)

    def step(self, action: int):
        self.step_count += 1
        self.set_action(action=action)
        for i in range(self.step_wait):
            self.model.step()
        truncated = (self.step_count >= self.max_step)
        obs = self.prey.get_observation()
        reward = self.reward_function(obs)
        return obs, reward, self.prey.finished, truncated, {}

    def reset(self):
        self.step_count = 0
        self.model.reset()
        obs = self.prey.get_observation()
        return obs, {}

    def render(self):
        if self.view is None:
            View(model=self.model)
        self.view.draw()
