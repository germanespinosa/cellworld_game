import random
import time

import pygame

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
             navigation=loader.navigation,
             actions=loader.full_action_list)

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
random_actions = 5



#predator
last_destination_time = 0

# prey
puff_cool_down_time = .5
puff_cool_down = 0
puff_threshold = .1
puffed = False
goal_location = (1.0, 0.5)
goal_threshold = .1


def render_puff_area(surface: pygame.Surface):
    predator_location = view.coordinate_converter.from_canonical(predator.state.location)
    puff_area_size = puff_threshold * view.coordinate_converter.screen_width
    puff_location = predator_location[0] - puff_area_size, predator_location[1] - puff_area_size
    puff_area_surface = pygame.Surface((puff_area_size * 2, puff_area_size * 2), pygame.SRCALPHA)
    puff_area_color = (255, 0, 0, 60) if puff_cool_down > 0 else (0, 0, 255, 60)
    pygame.draw.circle(puff_area_surface,
                       color=puff_area_color,
                       center=(puff_area_size, puff_area_size),
                       radius=puff_area_size)
    surface.blit(puff_area_surface,
                     puff_location)
    pygame.draw.circle(surface=surface,
                       color=(0, 0, 255),
                       center=predator_location,
                       radius=puff_area_size,
                       width=2)


view.add_render_step(render_puff_area)


while prey.running and not finished:
    print(prey.running)
    pre_observation = post_observation
    view.draw()
    if time.time() - last_action_time >= 3:
        if random_actions == 0:
            action_number = len(loader.full_action_list) - 1
        else:
            random_actions -= 1
            action_number = random.randint(0, len(loader.full_action_list) - 1)
        prey.set_action(action_number)
        last_action_time = time.time()
    delta_t = model.step()

    if puff_cool_down <= 0 and predator:
        predator_distance = distance(prey.state.location,
                                     predator.state.location)
        if predator_distance <= puff_threshold:
            if model.visibility.line_of_sight(prey.state.location, predator.state.location):
                puffed = True
                puff_cool_down = puff_cool_down_time

    if delta_t < puff_cool_down:
        puff_cool_down -= delta_t
    else:
        puff_cool_down = 0
    goal_distance = distance(goal_location, prey.state.location)
    if goal_distance <= goal_threshold:
        goal_achieved = True
        prey.running = False

    if last_destination_time + 1 < time.time():
        observation = predator.get_observation()
        last_destination_time = time.time()
        if "prey" in observation["agent_states"] and observation["agent_states"]["prey"]:
            predator.set_destination(observation["agent_states"]["prey"][0])

    if not predator.path:
        predator.set_destination(random.choice(loader.open_locations))


    print(prey.running)
    post_observation = prey.get_observation()
    t1 = time.time()
    print(1/(t1-t0))
    t0 = t1
