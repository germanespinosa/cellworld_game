import os.path

import cellworld as cw
from .model import Model


def save_log_output(model: Model,
                    experiment_name: str,
                    log_folder: str):

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    experiment = cw.Experiment(name=experiment_name,
                               world_configuration_name="hexagonal",
                               world_implementation_name="mice",
                               duration=0,
                               occlusions=model.world_name)

    frame = 0
    if hasattr(model, "puffed"):
        record_captures = True
    else:
        record_captures = False

    def after_reset(*args):
        nonlocal frame
        experiment.episodes.append(cw.Episode())
        frame = 0

    def after_stop(*args):
        import os
        output_file = os.path.join(log_folder, f"{experiment_name}.json")
        print(f"saving log file {output_file}")
        experiment.save(output_file)

    def after_step(*args):
        nonlocal frame
        episode: cw.Episode = experiment.episodes[-1]
        for agent_name, agent in model.agents.items():
            agent_step = cw.Step(time_stamp=model.time_step,
                                 location=cw.Location(*agent.state.location),
                                 rotation=90-agent.state.direction,
                                 agent_name=agent_name,
                                 frame=frame)
            episode.trajectories.append(agent_step)
        if record_captures:
            episode.captures.append(frame)
        frame += 1

    model.add_event_handler("after_reset", after_reset)
    model.add_event_handler("after_stop", after_stop)
    model.add_event_handler("after_step", after_step)
