"""
Trainer for the InfoGan
"""

from keras.callbacks import TensorBoard


class ModelTrainer(object):
    """
    ModelTrainer implements the training procedure of InfoGAN
    """

    def __init__(self, gan_model, data_generator, val_x):
        """__init__

        :param gan_model - the already initialized gan model
        :param data_generator - keras data generator
        """
        self.model = gan_model
        self.data_generator = data_generator

        self.board = TensorBoard(histogram_freq=100)
        self.board.set_model(self.model.disc_model)
        self.board.validation_data = [val_x]

    def train(self):
        epoch_count = 0
        for i in range(50):
            counter = 0
            for samples in self.data_generator():
                disc_loss = self.model.train_disc_pass(samples)
                counter += 1
                gen_losses = self.model.train_gen_pass()


                if counter % 100 == 0:
                    epoch_count += 1
                    print("Gen names: {}".format(self.model.enc_gen_model.metrics_names))
                    print("Gen losses: {}".format(gen_losses) + "\tDisc loss: {}".format(disc_loss))
                    loss_logs = {}
                    loss_logs['discriminator_loss'] = disc_loss
                    for loss, loss_name in zip(gen_losses, self.model.enc_gen_model.metrics_names):
                        loss_logs[loss_name] = loss
                    self.board.on_epoch_end(epoch_count, loss_logs)
                    epoch_count += 1
                    # self.model._sanity_check()
                    break
        self.board.on_train_end({})
