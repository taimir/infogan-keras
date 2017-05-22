import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from learn.data_management.interfaces import DataProvider


class SemiSupervisedSkeletonProvider(DataProvider):

    def __init__(self, batch_size, supervision=0.05):
        self.batch_size = batch_size
        self.supervision_frequency = int(1 / supervision)

    def _form_data(self, dir_path):
        sequences = []
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            frames = self._load_skeleton_file(file_path)

            sequences.append(frames)

        print("Sequences: {}".format(len(sequences)))
        sequence_lengths = [len(sequence) for sequence in sequences]
        print("Min / Max frames: {} - {}".format(min(sequence_lengths), max(sequence_lengths)))

        object_counts = []
        for sequence in sequences:
            object_counts += [len(frame) for frame in sequence]
        print("Min / Max objects: {} - {}".format(min(object_counts), max(object_counts)))

        joint_counts = []
        for sequence in sequences:
            for frame in sequence:
                joint_counts += [len(skeleton) for skeleton in frame]
        print("Min / Max joints: {} - {}".format(min(joint_counts), max(joint_counts)))

        assert min(joint_counts) == max(joint_counts), "Joint count must be fixed."

        max_objects = max(object_counts)
        numpy_sequences = []
        for sequence in sequences:
            numpy_frames = []
            for frame in sequence:
                if len(frame) < max_objects:
                    frame += [np.zeros(max(joint_counts), dtype="float32")] * \
                        (max_objects - len(frame))

                numpy_frame = np.concatenate(frame)
                numpy_frames.append(numpy_frame)

            numpy_sequences.append(np.stack(numpy_frames))

        data = pad_sequences(numpy_sequences, dtype="float32", padding="post")

        return data

    def _load_skeleton_file(self, file_path):
        """
        https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/read_skeleton_file.m
        """
        with open(file_path) as f:
            frames = []
            # number of frames
            frames_count = self._next_int(f)
            for i in range(frames_count):
                skeletons = []
                # number of skeletons in frame
                body_count = self._next_int(f)
                for j in range(body_count):
                    joints = []
                    # next row: in this order
                    # ID of the skeleton
                    # 6 binary 0/1s for: clippedEdges, handLeftConfidence, handLeftState,
                    # handRightConfidence, handRightState, isRestricted
                    # 2 floats: leanX and leanY
                    # integer: tracking state

                    # TODO: for now, I'll ignore that meta information about the skeleton
                    self._next_float_line(f)

                    # next line: joint count
                    joints_count = self._next_int(f)
                    for k in range(joints_count):
                        # next row: 11 floats for the joint info, in the following order:
                        # x, y, z coords
                        # depthX, depthY for matching with dRGB feed
                        # colorX, colorY for matching with RGB feed
                        # w, x, y, z orientation of the joint
                        # trackingState of the joint
                        joint_info = self._next_float_line(f)
                        # TODO: for now, ignore everything besides the 3D coords
                        coords = np.array(joint_info[:3], dtype="float32")

                        joints.append(coords)

                    skeleton = np.concatenate(joints)
                    skeletons.append(skeleton)

                frames.append(skeletons)

        return frames

    def _next_int(self, f):
        return int(next(f).split()[0])

    def _next_float_line(self, f):
        return [float(x) for x in next(f).split()]

    def iterate_minibatches(self):
        raise NotImplementedError

    def training_data(self):
        raise NotImplementedError

    def validation_data(self):
        raise NotImplementedError

    def test_data(self):
        raise NotImplementedError

if __name__ == "__main__":
    provider = SemiSupervisedSkeletonProvider(batch_size=100)
    provider._form_data(dir_path="/workspace/action_recogn/skeletons_data")
