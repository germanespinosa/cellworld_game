import random
from cellworld_game import BotEvade

bot_evade = BotEvade(world_name="21_05",
                     puff_cool_down_time=.5,
                     puff_threshold=.1,
                     goal_threshold=.05,
                     time_step=.025,
                     real_time=True,
                     render=True,
                     use_predator=True)

bot_evade.view.agent_perspective = 0

bot_evade.reset()

# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 50

action_count = len(bot_evade.loader.full_action_list)

while bot_evade.running:
    if bot_evade.time > last_destination_time + 2:
        if bot_evade.goal_achieved or random_actions == 0:
            destination = bot_evade.goal_location
            random_actions = 50
        else:
            random_actions -= 1
            destination = random.choice(bot_evade.loader.open_locations)
        bot_evade.prey.set_destination(destination)
        last_destination_time += 2
    bot_evade.step()
