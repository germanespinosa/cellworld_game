import random
import time
from cellworld_game import *
from cellworld_game.cellworld_loader import CellWorldLoader

loader = CellWorldLoader(world_name="21_05")

model = Model(arena=loader.arena,
              occlusions=loader.occlusions,
              time_step=.1,
              real_time=False)

navigation = Navigation(locations=loader.locations,
                        paths=loader.paths,
                        visibility=model.visibility)


predator = Robot(start_locations=loader.robot_start_locations,
                 open_locations=loader.open_locations,
                 navigation=navigation)

model.add_agent("predator", predator)


prey = Mouse(start_state=AgentState(location=(.05, .5),
                                    direction=0),
             goal_location=(1, .5),
             goal_threshold=.1,
             puff_threshold=.05,
             puff_cool_down_time=.5,
             navigation=navigation,
             actions=loader.full_action_list)

model.add_agent("prey", prey)

view = View(model=model)


model.reset()
post_observation = prey.get_observation()
last_action_time = time.time() - 3
while not prey.finished:
    pre_observation = post_observation
    print(pre_observation)
    view.draw()
    if time.time() - last_action_time >= 3:
        # decision
        action_number = random.randint(0, len(loader.full_action_list) - 1)
        prey.set_action(action_number)
        last_action_time = time.time()
    model.step()
    post_observation = prey.get_observation()
    # learning Gradient Descent

