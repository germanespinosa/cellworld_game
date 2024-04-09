from cellworld_game import Environment
from stable_baselines3 import PPO,DQN,SAC,DQN

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer


def random():
    def reward(observation):
        return -observation[6]
    env = Environment(world_name="21_05",
                      use_lppos=False,
                      use_predator=True,
                      max_step=200,
                      reward_function=reward)
    env.reset()

    for i in range(100000):
        if i%10000 == 0:
            print(i)
        print(i)
        obs, reward, done, tr, _ = env.step(env.action_space.sample())
        env.render()
        if i % 200 ==0:
            env.reset()

def DQN_train():
    def reward(observation):
        return -observation[6]

    env = Environment(world_name="21_05",
                      use_lppos=False,
                      use_predator=True,
                      max_step=300,
                      reward_function=reward)
    env.render()
    model = DQN("MlpPolicy",
                env,
                verbose=1,
                batch_size=256,
                learning_rate=1e-3,
                train_freq=(1, "step"),
                buffer_size=10000,
                learning_starts=1000,
                replay_buffer_class=ReplayBuffer,
                policy_kwargs={"net_arch": [256, 256]}
                )
    model.learn(total_timesteps=100000, log_interval=2)
    model.save("DQN")
    env.close()

def result_visualization():
    def reward(observation):
        return -observation[6]
    env = Environment(world_name="21_05",
                      use_lppos=False,
                      use_predator=True,
                      max_step=200,
                      reward_function=reward)
    loaded_model = DQN.load("DQN.zip")
    scores = []
    for i in range(100):
        obs,_ = env.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = env.step(action)
            score += reward
            env.render()
            # obs,_ = wrapped_env.reset()
        scores.append(score)
    env.close()



if __name__=="__main__":
    random()
    # DQN_train()
    # result_visualization()