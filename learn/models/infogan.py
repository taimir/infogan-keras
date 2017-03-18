"""
Implementation of the InfoGAN network
"""

import numpy as np
import keras.backend as K
from keras.layers import InputLayer, Dense, BatchNormalization, Activation, Merge
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
        disc_input = InputLayer(shape=self.image_shape)
        shared = discrimination_net(disc_input)

        shared_trunk = Model(inputs=[disc_input], outputs=shared)

        # binary output for the GAN classification
        disc_net = shared_trunk(disc_input)
        disc_out = Dense(1)(disc_net)

        def disc_loss(targets, preds):
            # targets are disregarded, as it's clear what they are
            targets_real = K.ones(shape=(self.batch_size))
            targets_fake = K.zeros(shape=(self.batch_size))
            targets = K.concatenate(targets_fake, targets_real, concat_axis=0)

            # the first output is the discriminator output
            return binary_crossentropy(targets, preds)

        # compile the disc. part only at first
        disc_model = Model(inputs=[disc_input], outputs=disc_out)
        disc_model.compile(optimizer='adam',
                           loss=disc_loss)
        self.disc_model = disc_model

        # GENERATOR
        # --------------------------------------------------------------------
        dummy_input = InputLayer(shape=(None,))
        latent_inputs = Lambda(function=self._sample_latents_inputs,
                               output_shape=self._get_latent_inputs_shape())(dummy_input)
        generation_params = generation_net(inputs=latent_inputs,
                                           image_shape=self.image_shape)

        generated = Lambda(function=self._sample_image,
                           output_shape=self.image_shape)(generation_params)

        # now append the real images to the generated ones
        real = InputLayer(shape=self.image_shape)
        merged = Merge([generated, real], mode='concat', concat_axis=0)

        # the discriminator output is needed
        disc_out = self.disc_model(merged)

        # the encoder produces the statistics of the p(c | x) distribution
        # which is a product of multiple factors (meaningful_dists)
        encoder_net = shared_trunk(merged)
        encoder_net = Dense(128)(encoder_net)
        encoder_net = BatchNormalization()(encoder_net)
        encoder_net = LeakyReLU()(encoder_net)

        # combine generator & discriminator
        # add outputs for the parameters of all assumed meaninful distributions
        dist_outputs = {}
        for name, dist in meaningful_dists.items():
            outputs = self._add_dist_outputs(dist, encoder_net)
            dist_outputs[name] = outputs

        outputs_list = [disc_out]
        for dist_output in dist_outputs.values():
            outputs_list += dist_output.values()
        encoder_out = K.concatenate(outputs_list, concat_axis=-1)

        gan_model = Model(inputs=[dummy_input], outputs=[disc_out, encoder_out])

        def gen_loss(targets, preds):
            # Again, the targets are essentially ignored
            param_index = 0
            sample_index = 0
            loss = 0
            # take only the generated samples
            preds = preds[:self.batch_size]

            # define the standard disc GAN loss
            loss = K.sum(-K.log(preds))

            for dist in self.meaningful_dists.values():
                # form the parameter dict of each distribution
                param_dict = {}
                for param_name, (dim, _) in dist.param_info():
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
                          loss=gen_loss)
        self.gan_model = gan_model

        # dummy targets, used during the training
        # this is a workaround for keras
        self.dummy_targets = np.ones((self.batch_size,))

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
        shapes = []
        for dist in self.meaningful_dists:
            shapes.append(dist.sample_size())
        for dist in self.noise_dists:
            shapes.append(dist.sample_size())
        return sum(shapes)

    def _sample_image(self, params):
        params_dict = {'p_vals': params}
        sampled_image = self.image_dist.sample(params_dict)
        return sampled_image

    def _add_dist_outputs(self, dist, layer):
        info = dist.params_info()
        outputs = {}
        for param, (dim, activation) in info.items():
            out = Dense(dim)(layer)
            out = Activation(activation)(out)
            outputs[param] = out
        return outputs

    def train_disc_pass(self, samples_batch):
        self.disc_model.train_on_batch(samples_batch, self.dummy_targets)

    def train_gen_pass(self, samples_batch):
        self.gan_model.train_on_batch(samples_batch, self.dummy_targets)
