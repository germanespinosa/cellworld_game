from cellworld_game import Environment
from stable_baselines3 import PPO,DQN,SAC,DQN

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer


def random(environment: Environment):

    environment.reset()

    for i in range(100000):
        if i % 10000 == 0:
            print(i)
        print(i)
        # obs, reward, done, tr, _ = environment.step(environment.action_space.sample())
        obs, reward, done, tr, _ = environment.step(280)
        environment.render()
        if i % 200 == 0:
            environment.reset()


def DQN_train(environment: Environment):

    environment.render()
    model = DQN("MlpPolicy",
                environment,
                verbose=1,
                batch_size=256,
                learning_rate=1e-4,
                train_freq=(1, "step"),
                buffer_size=400000,
                learning_starts=1000,
                replay_buffer_class=ReplayBuffer,
                policy_kwargs={"net_arch": [512, 512]}
                )
    model.learn(total_timesteps=400000, log_interval=2)
    model.save("DQN")
    env.close()


def result_visualization(environment: Environment):
    loaded_model = DQN.load("DQN.zip")
    scores = []
    for i in range(100):
        obs,_ = environment.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = environment.step(action)
            score += reward
            environment.render()
            # obs,_ = wrapped_env.reset()
        scores.append(score)
    environment.close()


def reward(observation):
    r = -observation[6]
    if observation[-1]:
        r += 1
    if observation[-3]:
        r -= 10
    return r


if __name__=="__main__":
    env = Environment(world_name="21_05",
                      use_lppos=False,
                      use_predator=True,
                      max_step=300,
                      reward_function=reward,
                      step_wait=20)
    #random(env)
    DQN_train(env)
    result_visualization(env)
