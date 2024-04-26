import random
from cellworld_game import DualEvade

bot_evade = DualEvade(world_name="21_05",
                      puff_cool_down_time=.5,
                      puff_threshold=.1,
                      goal_threshold=.05,
                      time_step=.025,
                      real_time=True,
                      render=True,
                      use_predator=True)


bot_evade.reset()

# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 10

action_count = len(bot_evade.loader.full_action_list)

while bot_evade.running:
    if bot_evade.time > last_destination_time + 3:
        if not bot_evade.prey_data_1.goal_achieved:
            if random_actions == 0:
                destination = bot_evade.goal_location
            else:
                random_actions -= 1
                destination = random.choice(bot_evade.loader.open_locations)
            bot_evade.prey_1.set_destination(destination)
        if not bot_evade.prey_data_2.goal_achieved:
            if random_actions == 0:
                destination = bot_evade.goal_location
            else:
                random_actions -= 1
                destination = random.choice(bot_evade.loader.open_locations)
            bot_evade.prey_2.set_destination(destination)
        last_destination_time += 3
    bot_evade.step()
