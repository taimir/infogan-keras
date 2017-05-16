import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

from learn.utils.visualization import image_grid
from learn.train.observers.interfaces import TrainingObserver


class InfoganTensorBoard(TrainingObserver):

    def __init__(self, model, experiment_dir, epoch_frequency, val_x=None, val_y=None):
        super(InfoganTensorBoard, self).__init__(model, epoch_frequency, val_x, val_y)

        self.logdir = experiment_dir
        self.sess = K.get_session()

        prior_params = self.model.prior.assemble_prior_params()
        self.board = TensorBoard(histogram_freq=20, log_dir=self.logdir)
        self.board.set_model(self.model.disc_train_model)

        self.batch_size = self.model.batch_size
        self.vis_data = [val_x[:self.batch_size]] + prior_params
        board_data = [val_x[:self.batch_size]]
        if self.model.encoder.supervised_dist:
            board_data += [val_y[:self.batch_size]]
        board_data += prior_params
        self.board.validation_data = board_data

        # feed disctionary for the images
        self.vis_feed_dict = dict(zip([self.model.real_input] +
                                      self.model.prior_param_inputs + [K.learning_phase()],
                                      self.vis_data + [0]))

        # add a generated images summary
        self.gen_image_summaries = list()
        self.real_enc_image_summaries = list()
        self.gen_enc_image_summaries = list()
        self.gen_cont_summaries_feeds = list()
        self.gen_cont_summaries = list()

        # initialize all tensorboard summaries
        self._init_gen_summaries()
        self._init_real_enc_summaries()
        self._init_gen_enc_summaries()
        self._init_gen_cont_summaries()

    def _init_gen_summaries(self):
        for dist_name, sampled_latent in self.model.sampled_latents.items():
            if "c1" in dist_name:
                sampled_class = K.argmax(sampled_latent, axis=1)
                for i in range(10):
                    which = tf.where(tf.equal(sampled_class, i))
                    # select only the images that were produced from c1 = i
                    selected = tf.gather(self.model.generated, which)
                    selected = tf.reshape(selected, (-1, 28, 28, 1))
                    grid = image_grid(input_tensor=selected,
                                      grid_shape=(5, 5),
                                      image_shape=(28, 28, 1))
                    summary = tf.summary.image("generated_from_c1={}".format(i),
                                               grid,
                                               max_outputs=1)
                    self.gen_image_summaries.append(summary)

    def _init_real_enc_summaries(self):
        # add Encoder results summaries
        for dist_name, stats in self.model.real_encodings.items():
            if "c1" == dist_name:
                for i in range(10):
                    output_class = K.argmax(stats["p_vals"], axis=1)
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
                    self.real_enc_image_summaries.append(summary)

    def _init_gen_enc_summaries(self):
        for dist_name, stats in self.model.gen_encodings.items():
            if "c1" == dist_name:
                for i in range(10):
                    output_class = K.argmax(stats["p_vals"], axis=1)
                    which = tf.where(tf.equal(output_class, i))

                    # select only the images that were encoded as c1 = i
                    selected = tf.gather(self.model.generated, which)
                    selected = tf.reshape(selected, (-1, 28, 28, 1))

                    grid = image_grid(input_tensor=selected,
                                      grid_shape=(5, 5),
                                      image_shape=(28, 28, 1))

                    summary = tf.summary.image("gen_encoded_as_c1={}".format(i),
                                               grid,
                                               max_outputs=1)
                    self.gen_enc_image_summaries.append(summary)

    def _init_gen_cont_summaries(self):
        # summaries for the continuous variables, covering the range from -1 to 1
        for i in range(10):
            selected = tf.reshape(self.model.generated, (-1, 28, 28, 1))
            grid = image_grid(input_tensor=selected,
                              grid_shape=(10, 10),
                              image_shape=(28, 28, 1))
            summary = tf.summary.image("gen_span_over_c2_c2_c1={}".format(i),
                                       grid,
                                       max_outputs=1)
            self.gen_cont_summaries.append(summary)

            feed_values = {
                "c1": to_categorical(np.array([i] * 100), num_classes=10),
                "c2": np.repeat(np.linspace(-1, 1, num=10).reshape((1, 10)), repeats=10, axis=0).reshape((100, 1)),
                "c3": np.repeat(np.linspace(-1, 1, num=10).reshape((10, 1)), repeats=10, axis=1).reshape((100, 1)),
                "z": np.zeros((100, 62)),
            }
            feed_dict = {K.learning_phase(): 0}

            for dist_name, sample_tensor in self.model.sampled_latents.items():
                feed_dict[sample_tensor] = feed_values[dist_name]

            self.gen_cont_summaries_feeds.append(feed_dict)

    def _update(self, epoch, epoch_results):
        # visualize images, generated and real, grouped by the c1 values

        for summary in self.gen_image_summaries:
            result = self.sess.run([summary],
                                   feed_dict=self.vis_feed_dict)
            summary_str = result[0]
            self.board.writer.add_summary(summary_str, epoch)

        for summary in self.gen_enc_image_summaries:
            result = self.sess.run([summary],
                                   feed_dict=self.vis_feed_dict)
            summary_str = result[0]
            self.board.writer.add_summary(summary_str, epoch)

        for summary in self.real_enc_image_summaries:
            result = self.sess.run([summary],
                                   feed_dict=self.vis_feed_dict)
            summary_str = result[0]
            self.board.writer.add_summary(summary_str, epoch)

        for summary, feed_dict in zip(self.gen_cont_summaries, self.gen_cont_summaries_feeds):
            result = self.sess.run([summary],
                                   feed_dict=feed_dict)
            summary_str = result[0]
            self.board.writer.add_summary(summary_str, epoch)

        loss_logs = epoch_results['losses']
        # plot the scalar values & histograms and flush other summaries
        self.board.on_epoch_end(epoch, loss_logs)

    def finish(self):
        self.board.on_train_end({})
