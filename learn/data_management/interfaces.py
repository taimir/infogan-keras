import abc


class DataProvider:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def iterate_minibatches():
        raise NotImplementedError

    @abc.abstractmethod
    def training_data():
        raise NotImplementedError

    @abc.abstractmethod
    def validation_data():
        raise NotImplementedError

    @abc.abstractmethod
    def test_data():
        raise NotImplementedError
