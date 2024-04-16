from cellworld_game import Reward
from cellworld_game import Environment
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

def create_vec_env(environment_count: int,
                   use_lppos: bool = True,
                   use_predator: bool = False,
                   max_steps: int = 300,
                   step_wait: int = 10,
                   reward_structure: dict = {},
                   **kwargs):

    return DummyVecEnv([lambda: Environment(world_name="21_05",
                                            use_lppos=use_lppos,
                                            use_predator=use_predator,
                                            max_step=max_steps,
                                            step_wait=step_wait,
                                            reward_function=Reward(reward_structure))
                        for _ in range(environment_count)])


