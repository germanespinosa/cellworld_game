import time
import cellworld as cw
from .view import View
from .model import Model
import numpy as np


def save_log_output(model: Model,
                    experiment_name: str,
                    log_file: str):

    experiment = cw.Experiment(name=experiment_name,
                               world_configuration_name="hexagonal",
                               world_implementation_name="mice",
                               duration=0,
                               occlusions=model.world_name)

    frame = 0

    def after_reset():
        nonlocal frame
        experiment.episodes.append(cw.Episode())
        frame = 0

    def after_stop():
        import os
        output_file = os.path.join(log_file, f"{experiment_name}.json")
        print(f"saving log file {output_file}")
        experiment.save(output_file)

    def after_step():
        nonlocal frame
        episode: cw.Episode = experiment.episodes[-1]
        for agent_name, agent in model.agents.items():
            agent_step = cw.Step(time_stamp=model.time_step,
                                 location=cw.Location(*agent.state.location),
                                 rotation=90-agent.state.direction,
                                 agent_name=agent_name,
                                 frame=frame)
            episode.trajectories.append(agent_step)
        frame += 1

    model.after_reset = after_reset
    model.after_stop = after_stop
    model.after_step = after_step
