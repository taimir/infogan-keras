"""
Trainer for the InfoGan
"""


class ModelTrainer(object):
    """
    ModelTrainer implements the training procedure of InfoGAN
    """

    def __init__(self, gan_model, data_generator):
        """__init__

        :param gan_model - the already initialized gan model
        :param data_generator - keras data generator
        """
        self.model = gan_model
        self.data_generator = data_generator

    def train(self):
        counter = 0
        for samples, _ in self.data_generator():
            self.model.train_disc_pass(samples)
            if counter % 10 == 9:
                self.model.train_gen_pass(samples)
