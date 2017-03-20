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

        self.board = TensorBoard(histogram_freq=100)
        self.board.set_model(self.model.enc_gen_model)

    def train(self):
        counter = 0
        epoch_count = 0
        for samples in self.data_generator():
            disc_loss = self.model.train_disc_pass(samples)
            counter += 1
            if counter % 2 == 1:
                gen_losses = self.model.train_gen_pass()
                # for i in range(len(self.model._layer_functions)):
                # name, output = self.model.activation(i, samples)

                print("Gen losses: {}".format(gen_losses) + "\tDisc loss: {}".format(disc_loss))
                loss_logs = {}
                loss_logs['discriminator_loss'] = disc_loss
                for loss, loss_name in zip(gen_losses, self.model.enc_gen_model.metrics_names):
                    loss_logs[loss_name] = loss
                self.board.on_epoch_end(epoch_count, loss_logs)
                epoch_count += 1

            if counter % 20 == 0:
                self.model._sanity_check()
        self.board.on_train_end({})
