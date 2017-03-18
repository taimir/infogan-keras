"""
Implementation of the InfoGAN network
"""

import keras.backend as K
from keras.layers import InputLayer, Dense, BatchNormalization, Activation, Merge
from keras.layers.core import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from learn.networks.convnets import discrimination_net, generation_net


class InfoGAN(object):
    """
    Puts together different networks to form the InfoGAN network as per:

    "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial
    Nets" by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
    """

    def __init__(self,
                 image_shape,
                 noise_dists,
                 meaningful_dists,
                 prior_params,
                 ):
        """__init__

        :param image_shape - triple (n_chan, img_height, img_width), shape of generated images
        :param noise_dists - dict of {'<name>': Distribution, ...}
        :param meaningful_dists - dict of {'<name>': Distribution, ...}
        :param prior_params - dict of {'<name>': <param_dict>,...}
        """

        self.image_shape = image_shape
        self.noise_dists = noise_dists
        self.meaningful_dists = meaningful_dists
        self.prior_params = prior_params

        # DISCRIMINATOR & ENCODER
        # --------------------------------------------------------------------
        # the encoder and discriminator share the same trunk
        # for now we define them as a separated model
        disc_input = InputLayer(shape=self.image_shape)
        disc_net = discrimination_net(disc_input)
        encoder_net = disc_net

        # binary output for the GAN classification
        disc_out = Dense(1)(disc_net)

        # the encoder produces the statistics of the p(c | x) distribution
        # which is a product of multiple factors (meaningful_dists)
        encoder_net = Dense(128)(encoder_net)
        encoder_net = BatchNormalization()(encoder_net)
        encoder_net = LeakyReLU()(encoder_net)

        # add outputs for the parameters of all assumed meaninful distributions
        dist_outputs = {}
        for name, dist in meaningful_dists.items():
            outputs = self._add_dist_outputs(dist, encoder_net)
            dist_outputs[name] = outputs

        outputs_list = [disc_out]
        for dist_output in dist_outputs.values():
            outputs_list += dist_output.values()

        # TODO: define the discriminator loss
        disc_loss = 0

        # compile the disc. part only at first
        disc_model = Model(inputs=[disc_input], outputs=outputs_list)
        disc_model.compile(optimizer='adam',
                           loss=disc_loss)
        self.disc_model = disc_model

        # GENERATOR
        # --------------------------------------------------------------------
        latent_inputs = Lambda(function=self._sample_inputs,
                               output_shape=self._get_latent_inputs_shape())
        generated = generation_net(inputs=latent_inputs, image_shape=image_shape)

        # now append the real images to the generated ones
        real = InputLayer(shape=self.image_shape)
        merged = Merge([generated, real], mode='concat', concat_axis=0)

        # combine generator & discriminator
        outputs_list = disc_model(merged)

        # TODO: define the generator loss
        gen_loss = 0

        # freeze the discriminator model during generator training
        disc_model.trainable = False
        gan_model = Model(inputs=[real], outputs=outputs_list)
        # TODO: make the discr. params untrainable before compilation
        gan_model.compile(optimizer='adam',
                          loss=gen_loss)
        self.gan_model = gan_model

    def _sample_latent_inputs(self):
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
            shapes.append(dist.sample_shape())
        for dist in self.noise_dists:
            shapes.append(dist.sample_shape())
        return sum(shapes)

    def _add_dist_outputs(self, dist, layer):
        info = dist.params_info()
        outputs = {}
        for param, (dim, activation) in info.items():
            out = Dense(dim)(layer)
            out = Activation(activation)(out)
            outputs[param] = out
        return outputs

    def train_pass(samples):
        raise NotImplementedError

    def test_pass(samples):
        raise NotImplementedError
