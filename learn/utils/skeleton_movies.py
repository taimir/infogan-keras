"""
http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/
"""
import os
import numpy as np

from moviepy.editor import VideoClip

from learn.data_management.skeleton_unsupervised import UnsupervisedSkeletonProvider
from learn.utils.opengl import setup_buffer, render_offscreen

CONNECT = np.array([2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14,
                    15, 1, 17, 18, 19, 2, 8, 8, 12, 12], dtype="int32") - 1

WIDTH = 300
HEIGHT = 400


def make_skeleton_movies(frame_sequences, movies_dir=".", fps=30):
    setup_buffer(WIDTH, HEIGHT)

    for i, frames in enumerate(frame_sequences):
        make_skeleton_animation(frames, i, fps, movies_dir)


def make_skeleton_animation(frames, index, fps, movies_dir):
    DEFAULT_FPS = 30.0

    def make_frame(t):
        frame_skeletons = frames[int(t * DEFAULT_FPS)]
        edges = []
        verticies = []
        for skeleton in frame_skeletons:
            verticies.append(skeleton)
            edges.append(np.array([[a, b] for a, b in zip(range(len(CONNECT)), CONNECT)]))

        verticies = np.concatenate(verticies)
        edges = np.concatenate(edges)

        arr = render_offscreen(verticies, edges, WIDTH, HEIGHT)

        return arr

    duration = frames.shape[0] / DEFAULT_FPS
    animation = VideoClip(make_frame, duration=duration)
    file_path = os.path.join(movies_dir, "sequence_{}.mp4".format(index))
    animation.write_videofile(file_path, fps=fps)

if __name__ == "__main__":
    provider = UnsupervisedSkeletonProvider(
        "/home/valor/workspace/infogan-keras/learn/utils", batch_size=100)
    data = provider._form_data(
        dir_path="/home/valor/workspace/infogan-keras/learn/utils", file_limit=10)

    make_skeleton_movies(data)
