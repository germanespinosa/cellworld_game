from cellworld_game import Environment


def reward(observation):
    return 1-observation[6] - 100 * observation[7]


env = Environment("21_05",
                  use_lppos=False,
                  use_predator=True,
                  max_step=200,
                  reward_function=reward,
                  step_wait=20)


for i in range(10000):
    env.reset()
    print()
    for s in range(200):
        print("%d " % s, end="")
        env.step(env.action_space.sample())
        if env.prey.finished:
            break
