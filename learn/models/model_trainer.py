"""
Trainer for the InfoGan
"""
import os

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

from learn.utils.visualization import image_grid


class ModelTrainer(object):
    """
    ModelTrainer implements the training procedure of InfoGAN
    """

    def __init__(self, gan_model, data_generator, val_x, val_y, experiment_id):
        """__init__

        :param gan_model - the already initialized gan model
        :param data_generator - keras data generator
        """
        self.model = gan_model
        self.data_generator = data_generator
        self.val_x = val_x
        self.val_y = val_y
        self.experiment_id = experiment_id

        self.board = TensorBoard(histogram_freq=20, log_dir=experiment_id)
        self.board.set_model(self.model.disc_train_model)

        prior_params = self.model._assemble_prior_params()
        self.vis_data = [val_x[:self.model.batch_size]] + prior_params
        self.board.validation_data = self.vis_data

        # feed disctionary for the images
        self.vis_feed_dict = dict(zip(self.model.disc_train_model.inputs + [K.learning_phase()],
                                      self.vis_data + [0]))

    def train(self, n_epochs=100):
        # add a generated images summary
        gen_image_summaries = []
        for dist_name, sampled_latent in self.model.sampled_latents.items():
            if "c1" in dist_name:
                sampled_class = K.argmax(sampled_latent, axis=1)
                for i in range(10):
                    which = tf.where(tf.equal(sampled_class, i))
                    # select only the images that were produced from c1 = i
                    selected = tf.gather(self.model.tensor_generated, which)
                    selected = tf.reshape(selected, (-1, 28, 28, 1))
                    grid = image_grid(input_tensor=selected,
                                      grid_shape=(5, 5),
                                      image_shape=(28, 28, 1))
                    summary = tf.summary.image("generated_from_c1={}".format(i),
                                               grid,
                                               max_outputs=1)
                    gen_image_summaries.append(summary)

        # add Encoder results summaries
        real_enc_image_summaries = []
        for output in self.model.c_post_outputs_real:
            if "c1" in output.name:
                for i in range(10):
                    output_class = K.argmax(output, axis=1)
                    which = tf.where(tf.equal(output_class, i))

                    # select only the images that were encoded as c1 = i
                    selected = tf.gather(self.model.real_input, which)
                    selected = tf.reshape(selected, (-1, 28, 28, 1))

                    grid = image_grid(input_tensor=selected,
                                      grid_shape=(5, 5),
                                      image_shape=(28, 28, 1))

                    summary = tf.summary.image("real_encoded_as_c1={}".format(i),
                                               grid,
                                               max_outputs=1)
                    real_enc_image_summaries.append(summary)

        gen_enc_image_summaries = []
        for output in self.model.c_post_outputs_gen:
            if "c1" in output.name:
                for i in range(10):
                    output_class = K.argmax(output, axis=1)
                    which = tf.where(tf.equal(output_class, i))

                    # select only the images that were encoded as c1 = i
                    selected = tf.gather(self.model.tensor_generated, which)
                    selected = tf.reshape(selected, (-1, 28, 28, 1))

                    grid = image_grid(input_tensor=selected,
                                      grid_shape=(5, 5),
                                      image_shape=(28, 28, 1))

                    summary = tf.summary.image("gen_encoded_as_c1={}".format(i),
                                               grid,
                                               max_outputs=1)
                    gen_enc_image_summaries.append(summary)

        # summaries for the continuous variables, covering the range from -1 to 1
        gen_cont_summaries = {}
        for output in self.model.c_post_outputs_gen:
            if "c1" not in output.name:
                selected = tf.reshape(self.model.tensor_generated, (-1, 28, 28, 1))
                grid = image_grid(input_tensor=selected,
                                  grid_shape=(10, 10),
                                  image_shape=(28, 28, 1))
                summary = tf.summary.image("gen_span_over_{}".format(output.name),
                                           grid,
                                           max_outputs=1)
                gen_cont_summaries[output.name] = summary

        cont_span_dict_c2 = {
            "c1": np.concatenate([to_categorical(np.array([i] * 10), num_classes=10)
                                  for i in range(10)], axis=0),
            "c2": np.repeat(np.linspace(-1, 1, num=10).reshape((1, 10)), repeats=10, axis=0).reshape((100, 1)),
            "c3": np.zeros((100, 1)),
            "z": np.zeros((100, 62)),
        }
        cont_span_dict_c3 = {
            "c1": np.concatenate([to_categorical(np.array([i] * 10), num_classes=10)
                                  for i in range(10)], axis=0),
            "c2": np.zeros((100, 1)),
            "c3": np.repeat(np.linspace(-1, 1, num=10).reshape((1, 10)), repeats=10, axis=0).reshape((100, 1)),
            "z": np.zeros((100, 62)),
        }

        cont_feed_dict_c2 = {K.learning_phase(): 0}
        cont_feed_dict_c3 = {K.learning_phase(): 0}
        for dist_name, sample_tensor in self.model.sampled_latents.items():
            cont_feed_dict_c2[sample_tensor] = cont_span_dict_c2[dist_name]
            cont_feed_dict_c3[sample_tensor] = cont_span_dict_c3[dist_name]

        # training iterations
        epoch_count = 0
        for i in range(n_epochs):
            counter = 0
            for samples in self.data_generator():
                disc_losses = self.model.train_disc_pass(samples)
                counter += 1
                gen_losses = self.model.train_gen_pass()

                if counter % 20 == 0:
                    print("Gen losses: {}".format(gen_losses))

                    loss_logs = {}
                    for loss, loss_name in zip(gen_losses, self.model.gen_train_model.metrics_names):
                        loss_logs["g_" + loss_name] = loss

                    for loss, loss_name in zip(disc_losses, self.model.disc_train_model.metrics_names):
                        loss_logs["d_" + loss_name] = loss

                    # visualize images, generated and real, grouped by the c1 values
                    sess = K.get_session()

                    for summary in gen_image_summaries:
                        result = sess.run([summary],
                                          feed_dict=self.vis_feed_dict)
                        summary_str = result[0]
                        self.board.writer.add_summary(summary_str, epoch_count)

                    for summary in gen_enc_image_summaries:
                        result = sess.run([summary],
                                          feed_dict=self.vis_feed_dict)
                        summary_str = result[0]
                        self.board.writer.add_summary(summary_str, epoch_count)

                    for summary in real_enc_image_summaries:
                        result = sess.run([summary],
                                          feed_dict=self.vis_feed_dict)
                        summary_str = result[0]
                        self.board.writer.add_summary(summary_str, epoch_count)

                    for cont_output_name, summary in gen_cont_summaries.items():
                        if "c2" in cont_output_name:
                            result = sess.run([summary],
                                              feed_dict=cont_feed_dict_c2)
                        else:
                            result = sess.run([summary],
                                              feed_dict=cont_feed_dict_c3)
                        summary_str = result[0]
                        self.board.writer.add_summary(summary_str, epoch_count)

                    # plot the scalar values & histograms and flush other summaries
                    self.board.on_epoch_end(epoch_count, loss_logs)

                    self.model.sanity_check()
                    epoch_count += 1

                if counter % 100 == 0:
                    # save the model weights
                    self.model.disc_train_model.save_weights(
                        os.path.join(self.experiment_id, "disc_train_model.hdf5"),
                        overwrite=True)
                    self.model.gen_train_model.save_weights(
                        os.path.join(self.experiment_id, "gen_train_model.hdf5"),
                        overwrite=True)
                    break

        self.board.on_train_end({})
