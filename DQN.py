import os
import json
import sys
import typing

from cellworld_game import Environment, Reward
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CustomMetricCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomMetricCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.captures = 0
        self.survival = 0
        self.m2 = 0
        self.m3 = 0

    def _on_step(self):
        self.captures += self.locals["infos"][0]["capture"]
        self.survival += self.locals["infos"][0]["survived"]
        return True

    def _on_rollout_end(self):
        self.logger.record('cellworld/captures', self.captures)
        self.logger.record('cellworld/survival', self.survival)

    def _on_rollout_start(self):
        pass

    def _on_episode_end(self):
        pass


def random(environment: Environment):
    environment.model.real_time = True
    environment.render_steps = True
    environment.reset()

    for i in range(100000):
        if i % 5 == 0:
            action = environment.action_space.sample()
        obs, reward, done, tr, _ = environment.step(action)
        if environment.prey.finished or i % 200 == 0:
            environment.reset()


def DQN_train(environment: Environment,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              batch_size: int,
              learning_starts: int,
              **kwargs: typing.Any):
    model = DQN("MlpPolicy",
                environment,
                verbose=1,
                batch_size=batch_size,
                learning_rate=learning_rate,
                train_freq=(1, "step"),
                buffer_size=training_steps,
                learning_starts=learning_starts,
                replay_buffer_class=ReplayBuffer,
                tensorboard_log="./tensorboard_logs/",
                policy_kwargs={"net_arch": network_architecture}
                )
    custom_callback = CustomMetricCallback()
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)
    env.close()


def result_visualization(environment: Environment,
                         name: str):
    environment.model.real_time = True
    environment.render_steps = True
    loaded_model = DQN.load("models/%s.zip" % name)
    scores = []
    for i in range(100):
        obs, _ = environment.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = environment.step(action)
            score += reward
        scores.append(score)
    environment.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-r":
            env = Environment(world_name="21_05",
                              use_lppos=True,
                              use_predator=True,
                              max_step=300,
                              step_wait=10)
            random(env)
            exit(0)
        else:
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
            else:
                model_name = input("Model Name: ")

            model_file = "models/%s_config.json" % model_name

            if os.path.exists(model_file):
                model_config = json.loads(open(model_file).read())

                if sys.argv[1] == "-t":
                    env = Environment(world_name="21_05",
                                      use_lppos=False,
                                      use_predator=True,
                                      max_step=300,
                                      step_wait=10,
                                      reward_function=Reward(model_config["reward_structure"]))
                    DQN_train(environment=env,
                              name="%s_control" % model_name,
                              **model_config)
                    env = Environment(world_name="21_05",
                                      use_lppos=True,
                                      use_predator=True,
                                      max_step=300,
                                      step_wait=10,
                                      reward_function=Reward(model_config["reward_structure"]))
                    DQN_train(environment=env,
                              name="%s_tlppo" % model_name,
                              **model_config)
                elif sys.argv[1] == "-v":
                    env = Environment(world_name="21_05",
                                      use_lppos=False,
                                      use_predator=True,
                                      max_step=300,
                                      step_wait=10)
                    result_visualization(environment=env,
                                         name="%s_tlppo" % model_name)
                exit(0)
            else:
                print("Model File not found")

    else:
        print("Missing parameters")

    print("Usage: python DQN.py <option> <model_name>")
    print("parameters:")
    print("  -t <model_name> : Executes training using the given model configuration file")
    print("  -v <model_name> : Shows visualization of the trained model")
    print("  -r : shows the environment with random policy")
    print()
    exit(1)
