from .view import View
from .model import Model
import pygame
import numpy as np


def save_video_output(model: Model,
                      video_folder: str):

    if model.view is None:
        raise ValueError("Model must have a view to save videos.")

    view: View = model.view
    view.gameplay_frames = []

    def before_stop():
        import os
        video_file = os.path.join(video_folder, f"episode_{model.episode_count:03}.mp4")
        print(f"saving video file {video_file}")
        if view.gameplay_frames:
            from moviepy.editor import ImageSequenceClip
            gameplay_clip = ImageSequenceClip(view.gameplay_frames, fps=int(1/model.time_step))
            gameplay_clip.write_videofile(filename=video_file,
                                          threads=16)
            view.gameplay_frames = []

    frame_count = 0

    def on_frame(surface, _):
        nonlocal frame_count
        print("FRAME:", frame_count)
        frame_count += 1
        frame = np.rot90(pygame.surfarray.array3d(surface))
        frame = np.flipud(frame)
        view.gameplay_frames.append(frame)

    view.on_frame = on_frame
    model.before_stop = before_stop
