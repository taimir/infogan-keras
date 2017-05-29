import keras.backend as K
import numpy as np
import os

from learn.train.observers.interfaces import TrainingObserver
from learn.utils.skeleton_movies import make_skeleton_movies


class SkeletonObserver(TrainingObserver):

    def __init__(self, model, epoch_frequency, movies_dir):
        super(SkeletonObserver, self).__init__(model, epoch_frequency, None, None)
        self.movies_dir = movies_dir

        self.sess = K.get_session()

        prior_params = self.model.prior.assemble_prior_params()
        self.feed_dict = dict(zip(self.model.prior_param_inputs + [K.learning_phase()],
                                  prior_params + [0]))

    def _update(self, epoch, epoch_results):
        if epoch % self.epoch_frequency != 0:
            return

        print("Generating skeleton movies under {}".format(self.movies_dir))
        # generated skeleton sequences
        skeleton_sequences = self.sess.run(self.model.generated, feed_dict=self.feed_dict)
        seq_shape = skeleton_sequences.shape
        skeleton_sequences = skeleton_sequences.reshape((seq_shape[0], seq_shape[1], -1, 25, 3))
        np.save(os.path.join(self.movies_dir, "gen_sequences.npy"), skeleton_sequences)
        # make_skeleton_movies(skeleton_sequences, movies_dir=self.movies_dir, fps=24)

    def finish(self):
        pass
