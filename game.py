import random
import time
from cellworld_game import *
from cellworld_game.cellworld_loader import CellWorldLoader

loader = CellWorldLoader(world_name="21_05")

model = Model(arena=loader.arena,
              occlusions=loader.occlusions,
              time_step=.025,
              real_time=True)

predator = Robot(start_locations=loader.robot_start_locations,
                 open_locations=loader.open_locations,
                 navigation=loader.navigation)

model.add_agent("predator", predator)


prey = Mouse(start_state=AgentState(location=(.05, .5),
                                    direction=0),
             goal_location_generator=lambda is_reset: (1, .5) if is_reset else None,
             goal_threshold=.1,
             puff_threshold=.1,
             puff_cool_down_time=.5,
             navigation=loader.navigation,
             actions=loader.full_action_list,
             predator=predator)

model.add_agent("prey", prey)

finished = False


def on_quit():
    global finished
    finished = True


view = View(model=model)
view.on_quit = on_quit

model.reset()
post_observation = prey.get_observation()
last_action_time = time.time() - 3
t0 = time.time()
while not prey.finished and not finished:
    pre_observation = post_observation
    view.draw()
    if time.time() - last_action_time >= 3:
        action_number = random.randint(0, len(loader.full_action_list) - 1)
        prey.set_action(action_number)
        last_action_time = time.time()
    model.step()
    post_observation = prey.get_observation()
    t1 = time.time()
    print(1/(t1-t0))
    t0 = t1

