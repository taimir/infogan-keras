"""
Trainer for the InfoGan
"""
import keras.backend as K
from keras.callbacks import TensorBoard
import tensorflow as tf


class ModelTrainer(object):
    """
    ModelTrainer implements the training procedure of InfoGAN
    """

    def __init__(self, gan_model, data_generator, val_x, val_y):
        """__init__

        :param gan_model - the already initialized gan model
        :param data_generator - keras data generator
        """
        self.model = gan_model
        self.data_generator = data_generator
        self.val_x = val_x
        self.val_y = val_y

        self.board = TensorBoard(histogram_freq=1)
        self.board.set_model(self.model.enc_gen_model)

        prior_params = self.model._assemble_prior_params()
        self.board.validation_data = prior_params

        # feed disctionary for the images
        self.gen_feed_dict = dict(zip(self.model.gen_model.inputs + [K.learning_phase()],
                                      prior_params + [0]))

    def train(self):
        # add a generated images summary
        gen_img_summary = tf.summary.image("generated",
                                           K.reshape(
                                               self.model.tensor_generated, (-1, 28, 28, 1)),
                                           max_outputs=9)

        real_summaries = []
        for i in range(10):
            real_img_summary = tf.summary.image("real_{}".format(i),
                                                K.reshape(self.model.real_input, (-1, 28, 28, 1)),
                                                max_outputs=9)
            real_summaries.append(real_img_summary)

        # add generated encodings summary
        enc_summaries = []
        for i in range(10):
            for output in self.model.posterior_outputs:
                if "c1" in output.name:
                    output_class = K.argmax(output, axis=1)
                    enc_summary = tf.summary.histogram("encoded_{}s".format(i),
                                                       output_class)
                    enc_summaries.append((i, enc_summary))

        # training iterations
        epoch_count = 0
        for i in range(50):
            counter = 0
            for samples in self.data_generator():
                disc_loss = self.model.train_disc_pass(samples)
                counter += 1
                gen_losses = self.model.train_gen_pass()

                if counter % 20 == 0:
                    print("Gen names: {}".format(self.model.enc_gen_model.metrics_names))
                    print("Gen losses: {}".format(gen_losses) + "\tDisc loss: {}".format(disc_loss))
                    loss_logs = {}
                    loss_logs['discriminator_loss'] = disc_loss
                    for loss, loss_name in zip(gen_losses, self.model.enc_gen_model.metrics_names):
                        loss_logs[loss_name] = loss

                    # visualize some generated images
                    sess = K.get_session()
                    result = sess.run([gen_img_summary], feed_dict=self.gen_feed_dict)
                    summary_str = result[0]
                    self.board.writer.add_summary(summary_str, epoch_count)

                    # visualize encoder predictions & real images
                    for i, summary in enc_summaries:
                        real_vals = self.val_x[self.val_y == i]
                        result = sess.run([summary],
                                          feed_dict={self.model.real_input: real_vals,
                                                     K.learning_phase(): 0})
                        summary_str = result[0]
                        self.board.writer.add_summary(summary_str, epoch_count)

                        # real images
                        real_img_summary = real_summaries[i]
                        result = sess.run([real_img_summary],
                                          feed_dict={self.model.real_input: real_vals,
                                                     K.learning_phase(): 0})
                        summary_str = result[0]
                        self.board.writer.add_summary(summary_str, epoch_count)

                    # plot the scalar values & histograms and flush other summaries
                    self.board.on_epoch_end(epoch_count, loss_logs)

                    # save the model weights
                    self.model.disc_model.save_weights("disc_model.hdf5", overwrite=True)
                    self.model.enc_gen_model.save_weights("enc_gen_model.hdf5", overwrite=True)

                    epoch_count += 1

                if counter % 300 == 0:
                    break

        self.board.on_train_end({})
