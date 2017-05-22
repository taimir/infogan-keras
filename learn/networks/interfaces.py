import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Network(object):

    def __init__(self):
        self.layers = []

    @abc.abstractmethod
    def apply(self, inputs):
        raise NotImplementedError

    def freeze(self):
        for layer in self.layers:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.layers:
            layer.trainable = True
