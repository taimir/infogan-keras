"""
Trainer for the InfoGan
"""
import numpy as np
import scipy.misc
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

        prior_params = self.model._assemble_prior_params()
        prior_params = [np.repeat(a=param[[0]], repeats=val_x.shape[0], axis=0) for param in prior_params]
        self.board.validation_data = [val_x] + prior_params

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

                    # save the model weights
                    self.model.disc_model.save_weights("disc_model.hdf5", overwrite=True)
                    self.model.enc_gen_model.save_weights("enc_gen_model.hdf5", overwrite=True)

                    images = self.model.generate()[:10, 0]
                    for i, image in enumerate(images):
                        scipy.misc.imsave("image_{}.png".format(i), image)

                    break
        self.board.on_train_end({})
