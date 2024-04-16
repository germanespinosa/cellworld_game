import os
import json
import sys
import typing

from vec_env import create_vec_env
from cellworld_game import Environment, Reward
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class CustomMetricCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomMetricCallback, self).__init__(verbose)
        self.captures_in_episode = 0
        self.episode_rewards = []
        self.captures = []
        self.captured = []
        self.survival = []
        self.finished = []
        self.truncated = []
        self.agents = {}

    def _on_step(self):
        if 'episode' in self.locals["infos"][0]:
            self.captures.append(self.locals["infos"][0]["captures"])
            if len(self.captures) > self.model._stats_window_size:
                self.captures.pop(0)

            self.survival.append(self.locals["infos"][0]["survived"])
            if len(self.survival) > self.model._stats_window_size:
                self.survival.pop(0)

            self.captured.append(1 if self.locals["infos"][0]["captures"] > 0 else 0)
            if len(self.captured) > self.model._stats_window_size:
                self.captured.pop(0)

            self.finished.append(0 if self.locals["infos"][0]["TimeLimit.truncated"] else 1)
            if len(self.finished) > self.model._stats_window_size:
                self.finished.pop(0)

            self.truncated.append(1 if self.locals["infos"][0]["TimeLimit.truncated"] else 0)
            if len(self.truncated) > self.model._stats_window_size:
                self.truncated.pop(0)

            self.logger.record('cellworld/avg_captures', safe_mean(self.captures))
            self.logger.record('cellworld/survival_rate', safe_mean(self.survival))
            self.logger.record('cellworld/ep_finished', sum(self.finished))
            self.logger.record('cellworld/ep_truncated', sum(self.truncated))
            self.logger.record('cellworld/ep_captured', sum(self.captured))


            for agent_name, agent_stats in self.locals["infos"][0]["agents"].items():
                if agent_name not in self.agents:
                    self.agents[agent_name] = {}
                for stat, value in agent_stats.items():
                    if stat not in self.agents[agent_name]:
                        self.agents[agent_name][stat] = []
                    self.agents[agent_name][stat].append(value)
                    if len(self.agents[agent_name][stat]) > self.model._stats_window_size:
                        self.agents[agent_name][stat].pop(0)
                    stat_values = self.agents[agent_name][stat]
                    self.logger.record('cellworld/{}_{}'.format(agent_name, stat), safe_mean(stat_values))

        return True

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


def DQN_train(environment,
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
                    vec_envs = create_vec_env(environment_count=10,
                                              world_name="21_05",
                                              use_lppos=False,
                                              use_predator=True,
                                              max_step=300,
                                              step_wait=10,
                                              reward_structure = model_config["reward_structure"])

                    DQN_train(environment=vec_envs,
                              name="%s_control" % model_name,
                              **model_config)

                    vec_envs = create_vec_env(environment_count=10,
                                              world_name="21_05",
                                              use_lppos=True,
                                              use_predator=True,
                                              max_step=300,
                                              step_wait=10,
                                              reward_structure=model_config["reward_structure"])

                    DQN_train(environment=vec_envs,
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
