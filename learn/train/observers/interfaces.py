import abc


class TrainingObserver:

    __metaclass__ = abc.ABCMeta

    def __init__(self, model, epoch_frequency, val_x=None, val_y=None):
        """__init__

        :param model - model which is being trained and observed
        :param epoch_frequency - how often should the observer actuallyd do something
        :param val_x - (optional) validation samples
        :param val_y - (optional) validation labels
        """
        self.model = model
        self.val_x = val_x
        self.val_y = val_y
        self.epoch_frequency = epoch_frequency

    def update(self, epoch, epoch_results):
        if epoch % self.epoch_frequency == 0:
            self._update(epoch, epoch_results)

    @abc.abstractmethod
    def _update(self, epoch, epoch_results):
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self):
        raise NotImplementedError
