import sys
from cellworld_game import Environment
from stable_baselines3 import DQN

from stable_baselines3.common.buffers import ReplayBuffer


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
              num_steps: int):
    # environment.render()
    model = DQN("MlpPolicy",
                environment,
                verbose=1,
                batch_size=256,
                learning_rate=1e-4,
                train_freq=(1, "step"),
                buffer_size=500000,
                learning_starts=3000,
                replay_buffer_class=ReplayBuffer,
                tensorboard_log="./tensorboard_logs/",
                policy_kwargs={"net_arch": [512, 512]}
                )
    model.learn(total_timesteps=num_steps, log_interval=10, tb_log_name=name)
    model.save(name)
    env.close()


def result_visualization(environment: Environment,
                         name: str):
    environment.model.real_time = True
    environment.render_steps = True
    loaded_model = DQN.load("%s.zip" % name)
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


def reward(observation):
    r = 0
    if observation[-1]:
        r += 1
    if observation[-3]:
        r -= 100
    return r


if __name__ == "__main__":
    env = Environment(world_name="21_05",
                      use_lppos=True,
                      use_predator=True,
                      max_step=300,
                      reward_function=reward,
                      step_wait=10)

    if sys.argv[1] == "-r":
        random(env)
    else:
        model_name = input("Model Name: ")
        if sys.argv[1] == "-t":
            DQN_train(environment=env,
                      name=model_name,
                      num_steps=500000)
        else:
            result_visualization(environment=env,
                                 name=model_name)
