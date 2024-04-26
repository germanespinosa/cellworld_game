import random
from cellworld_game import Oasis

bot_evade = Oasis(world_name="oasis_island7_02",
                  goal_locations=[(0.265625, 0.5), (0.3125, 0.7435696448143734), (0.3125, 0.1752404735808355), (0.4765625, 0.45940505919760444), (0.640625, 0.7435696448143734), (0.6875, 0.1752404735808355), (0.78125, 0.5)],
                  puff_cool_down_time=.5,
                  puff_threshold=.1,
                  goal_threshold=.025,
                  time_step=.025,
                  real_time=True,
                  render=True,
                  use_predator=True)


bot_evade.reset()

# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 5

action_count = len(bot_evade.loader.full_action_list)

while bot_evade.running:
    if bot_evade.time > last_destination_time + 2:
        if bot_evade.goal_achieved or random_actions == 0:
            destination = bot_evade.goal_location
            random_actions = 5
        else:
            random_actions -= 1
            destination = random.choice(bot_evade.loader.open_locations)
        bot_evade.prey.set_destination(destination)
        last_destination_time += 2
    bot_evade.step()
