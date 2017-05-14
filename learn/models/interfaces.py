import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Model:

    @abc.abstractmethod
    def train_on_minibatch(self, samples, labels):
        raise NotImplementedError
