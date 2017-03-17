"""
Implementation of the InfoGAN network
"""

from keras.layers import InputLayer, Dense, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from learn.networks.convnets import discrimination_net, generation_net

class InfoGAN(object):
    """
    Puts together different networks to form the InfoGAN network as per:

    "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial
    Nets" by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
    """

    def __init__(object, image_shape=(1, 28, 28)):
        # TODO: define the input and output dims properly
        output_latent_dims = 12
        input_latent_dims = 124 + 12

        # TODO: define the proper z and c inputs
        inputs = InputLayer(shape=(input_latent_dims,))

        # GENERATOR
        generated = generation_net(inputs=inputs, image_shape=image_shape)

        # DISCRIMINATOR & ENCODER
        # the encoder and discriminator share the same trunk
        disc_net = discrimination_net(generated)
        encoder_net = disc_net

        # binary output for the GAN classification
        disc_net = Dense(1)(disc_net)

        # the encoder produces the statistics of the p(c | x) distribution
        encoder_net = Dense(128)(encoder_net)
        encoder_net = BatchNormalization()(encoder_net)
        encoder_net = LeakyReLU()(encoder_net)
        encoder_net = Dense(output_latent_dims)


        # TODO: now define the training objectives (two losses) of the models

    def train_pass(samples):
        raise NotImplementedError

    def test_pass(samples):
        raise NotImplementedError
