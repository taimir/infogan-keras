import abc


class Model:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train_on_minibatch(self, samples, labels):
        raise NotImplementedError
