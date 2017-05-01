"""
Trainer for the InfoGan
"""


class ModelTrainer:

    def __init__(self,
                 model,
                 data_provider,
                 observers):
        """__init__

        :param model - a model that should be trained
        :param data_provider - data provider that can iterate over minibatches of data
        :observers - list of TrainingObserver inhereting classes, which supplement the
            training procedure.
        """
        self.model = model
        self.data_provider = data_provider
        self.observers = observers

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            for minibatch in self.data_provider.iterate_minibatches():
                artifacts = self.model.train_on_minibatch(*minibatch)

            for observer in self.observers:
                observer.update(epoch, artifacts)

        for observer in self.observers:
            observer.finish()
