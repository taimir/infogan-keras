"""
Trainer for the InfoGan
"""


class ModelTrainer(object):
    """
    ModelTrainer implements the training procedure of InfoGAN
    """

    def __init__(self, model):
        self.model = model

    def train(self, samples):
        raise NotImplementedError
