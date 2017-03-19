"""
Trainer for the InfoGan
"""

from keras.callbacks import TensorBoard


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

        self.board = TensorBoard()
        self.board.set_model(self.model)

    def train(self):
        counter = 0
        epoch_count = 0
        for samples in self.data_generator():
            disc_loss = self.model.train_disc_pass(samples)
            counter += 1
            if counter % 10 == 9:
                gen_losses = self.model.train_gen_pass(samples)
                for i in range(len(self.model._layer_functions)):
                    name, output = self.model.activation(i, samples)

                    continue

                print("Gen losses: {}".format(gen_losses) + "\tDisc loss: {}".format(disc_loss))
                loss_logs = {}
                loss_logs['disc'] = disc_loss
                loss_logs['gen_adversarial'] = gen_losses[0]
                loss_logs['gen_MI'] = gen_losses[1]
                self.board.on_epoch_end(epoch_count, loss_logs)
                epoch_count += 1
        self.board.on_train_end({})
