import sys
from cellworld_game import Environment
from stable_baselines3 import PPO,DQN,SAC,DQN

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer


def random(environment: Environment):
    environment.model.real_time = True
    environment.render_steps = True
    environment.reset()

    for i in range(100000):
        if i % 10000 == 0:
            print(i)
        print(i)
        obs, reward, done, tr, _ = environment.step(environment.action_space.sample())
        #obs, reward, done, tr, _ = environment.step(len(environment.loader.full_action_list)-1)
        if environment.prey.finished or i % 200 == 0:
            environment.reset()


def DQN_train(environment: Environment):
    # environment.render()
    model = DQN("MlpPolicy",
                environment,
                verbose=1,
                batch_size=256,
                learning_rate=5e-6,
                train_freq=(1, "step"),
                buffer_size=500000,
                learning_starts=3000,
                replay_buffer_class=ReplayBuffer,
                policy_kwargs={"net_arch": [512, 512]}
                )
    model.learn(total_timesteps=500000, log_interval=10)
    model.save("DQN")
    env.close()


def result_visualization(environment: Environment):
    environment.model.real_time = True
    environment.render_steps = True
    loaded_model = DQN.load("DQN.zip")
    scores = []
    for i in range(100):
        obs,_ = environment.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = environment.step(action)
            score += reward
            # obs,_ = wrapped_env.reset()
        scores.append(score)
    environment.close()


def reward(observation):
    r = 0
    # r -= observation[6]
    if observation[-1]:
        r += 10
    if observation[-3]:
        r -= 100
    return r


if __name__ == "__main__":
    env = Environment(world_name="21_05",
                      use_lppos=False,
                      use_predator=True,
                      max_step=300,
                      reward_function=reward,
                      step_wait=10)
    print(len(env.loader.full_action_list)-1)
    if sys.argv[1] == "-r":
        random(env)
    elif sys.argv[1] == "-t":
        DQN_train(env)
    else:
        result_visualization(env)
