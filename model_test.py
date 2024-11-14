import torch
import cellworld_belief as cb
import cellworld_game as cg
import datetime
import numpy as np



world_name = "oasis_island7_02"
loader = cg.CellWorldLoader(world_name=world_name)
model = cg.Model(render=True,
                 world_name=world_name,
                 arena=loader.arena,
                 occlusions=loader.occlusions,
                 time_step=1/90)

cg.save_video_output(video_folder=".", model=model)

mouse = cg.Mouse(start_state=cg.AgentState(location=(.05, .5),direction=0),
                 navigation=loader.navigation)

robot = cg.Robot(start_locations=loader.robot_start_locations,
                 open_locations=loader.open_locations,
                 navigation=loader.navigation)
robot.visible = False
model.add_agent(agent=mouse, name="prey")
model.add_agent(agent=robot, name="predator")
model.reset()
frames = 0
time = datetime.datetime.now()

bs_mouse = cb.BeliefState(model=model,
                          definition=50,
                          components=[],
                          agent_name="prey",
                          other_name="predator",
                          probability=0,
                          color=(0, 0, 255))

# bs_robot = cb.BeliefState(model=model,
#                           definition=50,
#                           components=[],
#                           agent_name="prey",
#                           other_name="predator",
#                           probability=0,
#                           color=(255, 0, 0))

for i in range(100):
    frames += 1
    # set the mouse location (and orientation if you have it)
    mouse.state.location = mouse.state.location[0] + .001, mouse.state.location[1]

    # set the probability matrix for the belief state
    bs_mouse.probability_distribution = torch.from_numpy(np.random.random(size=(86, 100))).to(bs_mouse.device)
    # bs_robot.probability_distribution = torch.from_numpy(np.random.random(size=(86, 100))).to(bs_robot.device)
    # bs.probability_distribution = torch.zeros(size=(86, 100)).to(bs.device)
    # bs.probability_distribution = torch.ones(size=(86, 100)).to(bs.device)

    # renders
    model.view.render()

    if frames % 100 == 0:
        print(100 / (datetime.datetime.now() - time).total_seconds())
        time = datetime.datetime.now()

model.stop()
