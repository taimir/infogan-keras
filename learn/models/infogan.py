"""
Implementation of the InfoGAN network
"""

import numpy as np
import keras.backend as K
from keras.layers import Input,  Dense, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.objectives import binary_crossentropy

from learn.networks.convnets import discrimination_net, generation_net


class InfoGAN(object):
    """
    Puts together different networks to form the InfoGAN network as per:

    "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial
    Nets" by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
    """

    def __init__(self,
                 batch_size,
                 image_shape,
                 noise_dists,
                 meaningful_dists,
                 image_dist,
                 prior_params,
                 ):
        """__init__

        :param batch_size - number of real samples passed at each iteration
        :param image_shape - triple (n_chan, img_height, img_width), shape of generated images
        :param noise_dists - dict of {'<name>': Distribution, ...}
        :param meaningful_dists - dict of {'<name>': Distribution, ...}
        :param image_dist - Distribution of the image, for sampling after the generator
        :param prior_params - dict of {'<name>': <param_dict>,...}
        """

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.noise_dists = noise_dists
        self.meaningful_dists = meaningful_dists
        self.image_dist = image_dist
        self.prior_params = prior_params

        # DISCRIMINATOR & ENCODER
        # --------------------------------------------------------------------
        # the encoder and discriminator share the same trunk
        # for now we define them as a separated model
        disc_input = Input(shape=self.image_shape, name="d_real_input")
        shared = discrimination_net(disc_input)

        shared_trunk = Model(inputs=[disc_input], outputs=shared, name="shared_trunk_d_e")

        # binary output for the GAN classification
        disc_net = shared_trunk(disc_input)
        disc_out = Dense(1, name="d_classif_layer")(disc_net)

        def disc_loss(targets, preds):
            # targets are disregarded, as it's clear what they are
            targets_real = K.ones(shape=(self.batch_size, 1))
            targets_fake = K.zeros(shape=(self.batch_size, 1))
            targets = K.concatenate([targets_fake, targets_real], axis=0)

            # the first output is the discriminator output
            return binary_crossentropy(targets, preds)

        # compile the disc. part only at first
        disc_model = Model(inputs=[disc_input], outputs=disc_out, name="disc_model")
        disc_model.compile(optimizer='adam',
                           loss=disc_loss)
        self.disc_model = disc_model

        # GENERATOR
        # --------------------------------------------------------------------
        dummy_input = Input(shape=(1,), name="g_dummy_input")
        latent_inputs = Lambda(function=self._sample_latent_inputs,
                               output_shape=self._get_latent_inputs_shape(),
                               name="g_latent_sampling")(dummy_input)
        generation_params = generation_net(inputs=latent_inputs,
                                           image_shape=self.image_shape)

        generated = Lambda(function=self._sample_image,
                           output_shape=self.image_shape,
                           name="g_image_sampling")(generation_params)

        self.gen_model = Model(inputs=[dummy_input], outputs=[generated])
        # NOTE: the loss here does not matter, it won't be used ...
        # the model is just compiled so that we can generate samples from it
        self.gen_model.compile(optimizer='adam', loss='binary_crossentropy')

        # now append the real images to the generated ones
        real = Input(shape=self.image_shape, name="g_real_input")
        merged = Concatenate(axis=0, name="g_concat_fake_real")([generated, real])

        # the discriminator output is needed
        disc_out = self.disc_model(merged)

        # the encoder produces the statistics of the p(c | x) distribution
        # which is a product of multiple factors (meaningful_dists)
        encoder_net = shared_trunk(merged)
        encoder_net = Dense(128, name="e_dense_1")(encoder_net)
        encoder_net = BatchNormalization(name="e_dense_bn_1")(encoder_net)
        encoder_net = LeakyReLU(name="e_dense_activ_1")(encoder_net)

        # combine generator & discriminator
        # add outputs for the parameters of all assumed meaninful distributions
        dist_outputs = {}
        for name, dist in meaningful_dists.items():
            outputs = self._add_dist_outputs(name, dist, encoder_net)
            dist_outputs[name] = outputs

        latent_outputs_list = []
        for dist_output in dist_outputs.values():
            latent_outputs_list += dist_output.values()
        encoder_out = Concatenate(axis=-1, name="e_output_concat")(latent_outputs_list)

        gan_model = Model(inputs=[dummy_input, real], outputs=[disc_out, encoder_out],
                          name="full_GAN_model")

        def gan_loss(targets, preds):
            # again, targets are not needed - ignore them
            # take only the generated samples
            preds = preds[:self.batch_size]

            # define the standard disc GAN loss
            loss = K.sum(-K.log(preds))

            return loss

        # TODO: instead of indexing 1 big loss, make a dictionary of one loss
        # per distribution
        def mi_loss(targets, preds):
            # Again, the targets are essentially ignored
            param_index = 0
            sample_index = 0
            loss = 0

            # take only the generated samples
            preds = preds[:self.batch_size]

            for dist in self.meaningful_dists.values():
                # form the parameter dict of each distribution
                param_dict = {}
                for param_name, (dim, _) in dist.param_info().items():
                    param_dict[param_name] = preds[:, param_index:param_index + dim]
                    param_index += dim

                # now get the sampled latent factors for this dist.
                sample_size = dist.sample_size()
                latent_samples = latent_inputs[:, sample_index:sample_index + sample_size]
                sample_index += sample_size

                # finally compute the NLL
                loss += dist.nll(latent_samples, param_dict)

            return loss

        # freeze the discriminator model during generator training
        disc_model.trainable = False
        gan_model.compile(optimizer='adam',
                          loss=[gan_loss, mi_loss])
        self.gan_model = gan_model

        # dummy targets, used during the training
        # this is a workaround for keras

    def _sample_latent_inputs(self, dummy_input):
        noise_samples = []
        for name, dist in self.noise_dists.items():
            params_dict = self.prior_params[name]
            noise_samples.append(dist.sample(params_dict))

        salient_samples = []
        for name, dist in self.meaningful_dists.items():
            params_dict = self.prior_params[name]
            salient_samples.append(dist.sample(params_dict))

        # put them all together, salient come first
        return K.concatenate(salient_samples + noise_samples, axis=-1)

    def _get_latent_inputs_shape(self):
        sizes = []
        for dist in self.meaningful_dists.values():
            sizes.append(dist.sample_size())
        for dist in self.noise_dists.values():
            sizes.append(dist.sample_size())
        return (sum(sizes),)

    def _sample_image(self, params):
        params_dict = {'p': params}
        sampled_image = self.image_dist.sample(params_dict)
        return sampled_image

    def _add_dist_outputs(self, dist_name, dist, layer):
        info = dist.param_info()
        outputs = {}
        for param, (dim, activation) in info.items():
            out = Dense(dim, name="e_dense_{}_{}".format(dist_name, param))(layer)
            out = Activation(activation, name="e_activ_{}_{}".format(dist_name, param))(out)
            outputs[param] = out
        return outputs

    def train_disc_pass(self, samples_batch):
        fake_x = self.generate()
        batch = np.concatenate([fake_x, samples_batch], axis=0)
        dummy_targets = np.ones((2 * self.batch_size,))

        self.disc_model.train_on_batch(batch, dummy_targets)

    def train_gen_pass(self, samples_batch):
        dummy_inputs = np.ones((self.batch_size, 1))
        dummy_targets = np.ones((self.batch_size,))
        self.gan_model.train_on_batch([dummy_inputs, samples_batch], [dummy_targets, dummy_targets])

    def generate(self):
        dummy = np.ones((self.batch_size, 1))
        return self.gen_model.predict(dummy, batch_size=self.batch_size)
