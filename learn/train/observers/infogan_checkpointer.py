import os
from learn.train.observers.interfaces import TrainingObserver


class InfoganCheckpointer(TrainingObserver):

    def __init__(self, model, experiment_dir, epoch_frequency):
        self.checkpoint_dir = experiment_dir

        super(InfoganCheckpointer, self).__init__(model, epoch_frequency, None, None)

    def _update(self, epoch, epoch_results):
        # save the model weights
        self.model.disc_train_model.save_weights(
            os.path.join(self.checkpoint_dir, "disc_train_model.hdf5"),
            overwrite=True)
        self.model.gen_train_model.save_weights(
            os.path.join(self.checkpoint_dir, "gen_train_model.hdf5"),
            overwrite=True)

    def finish(self):
        pass
