import random
from cellworld_game import BotEvade, save_video_output, save_log_output, Agent

bot_evade = BotEvade(world_name="00_00",
                     puff_cool_down_time=.5,
                     puff_threshold=.1,
                     goal_threshold=.05,
                     time_step=.025,
                     real_time=False,
                     render=True,
                     use_predator=True,
                     predator_prey_forward_speed_ratio=1.5,
                     predator_prey_turning_speed_ratio=1.5)
                     # point_of_view=BotEvade.PointOfView.PREY,
                     # agent_render_mode=Agent.RenderMode.POLYGON)

save_video_output(bot_evade, "videos")
save_log_output(bot_evade, "test", "logs")
bot_evade.reset()
bot_evade.view.show_sprites = False

# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 10

action_count = len(bot_evade.loader.full_action_list)

while bot_evade.running:
    if bot_evade.time > last_destination_time + 2:
        if bot_evade.prey_data.goal_achieved or random_actions == 0:
            destination = bot_evade.goal_location
            random_actions = 10
        else:
            random_actions -= 1
            destination = random.choice(bot_evade.loader.open_locations)
        bot_evade.prey.set_destination(destination)
        last_destination_time += 2
    bot_evade.step()
