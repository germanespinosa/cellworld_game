import time
from cellworld_game import *
import cellworld as cw
import math
import pygame

world = cw.World.get_from_parameters_names("hexagonal", "canonical", "21_05")
model = Model(world=world, time_step=.5, real_time=True)
start_locations = [tuple(c.location.get_values()) for c in world.cells.free_cells()]
predator = Robot(start_locations)
predator.state.direction = math.pi
predator.state.location = (.8, .5)

model.add_agent("predator", predator)
model.agents_dynamics["predator"].forward_speed = .01
model.agents_dynamics["predator"].turn_speed = .001

prey = Mouse()
prey.state.direction = math.pi
prey.state.location = (.3, .5)
model.add_agent("prey", prey)
model.agents_dynamics["prey"].turn_speed = 0
model.agents_dynamics["prey"].forward_speed = .001

view = View(model=model)
model.reset()

# screen = pygame.display.set_mode((900, 300))
# ray_tracing = RayTracing(model=model, horizontal_view_field=270, vertical_view_field=90, resolution=5)
while True:
    time_starts = time.time()
    # ray_tracing.render(perspective=prey.state,
    #                    screen=screen)
    view.render()
    model.step()
    print(1/(time.time() - time_starts))
