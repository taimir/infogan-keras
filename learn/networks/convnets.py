"""
Some network structures used by the InfoGAN
"""

import abc

import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Flatten, \
    Reshape
from keras.layers.advanced_activations import LeakyReLU


class Network(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.layers = []

    def apply(self, inputs):
        network = inputs
        for layer in self.layers:
            network = layer(network)
        return network

    def freeze(self):
        for layer in self.layers:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.layers:
            layer.trainable = True


class GeneratorNet(Network):

    def __init__(self, image_shape):
        self.layers = []

        # a fully connected is needed to bring the inputs to a shape suitable for convolutions
        self.layers.append(Dense(units=128, name="g_dense_1"))
        self.layers.append(BatchNormalization(name="g_dense_bn_1", axis=-1))
        self.layers.append(Activation(activation=K.relu, name="g_dense_activ_1"))

        self.layers.append(Dense(units=image_shape[1] // 4 * image_shape[2] // 4 * 32,
                                 name="g_dense_2"))
        self.layers.append(BatchNormalization(name="g_dense_bn_2", axis=-1))
        self.layers.append(Activation(activation=K.relu, name="g_dense_activ_2"))

        # # # I use the `th` orientation of theano
        self.layers.append(Reshape(target_shape=(32, image_shape[1] // 4, image_shape[2] // 4),
                                   name="g_reshape"))

        # # start applying the deconv layers
        self.layers.append(Conv2DTranspose(filters=16, kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           data_format='channels_first',
                                           name="g_deconv_1"))
        self.layers.append(BatchNormalization(name="g_deconv_bn_1", axis=1))
        self.layers.append(Activation(activation=K.relu, name="g_deconv_activ_1"))

        # # TODO: if we'll be generating color images, this needs to produce
        # # a 256 * image_shape[0] number of channels
        self.layers.append(Conv2DTranspose(filters=image_shape[0], kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           data_format='channels_first',
                                           name="g_deconv_2"))
        self.layers.append(Activation(activation=K.sigmoid, name="g_deconv_activ_2"))


class SharedNet(Network):

    def __init__(self):
        self.layers = []

        self.layers.append(Conv2D(filters=16,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  name="d_conv_1"))
        self.layers.append(LeakyReLU(name="d_conv_activ_1"))

        self.layers.append(Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  name="d_conv_2"))
        self.layers.append(BatchNormalization(name="d_conv_bn_2", axis=1))
        self.layers.append(LeakyReLU(name="d_conv_activ_2"))

        self.layers.append(Flatten(name="d_flatten"))
        self.layers.append(Dense(units=128, name="d_dense_1"))
        self.layers.append(BatchNormalization(name="d_dense_bn_1", axis=-1))
        self.layers.append(LeakyReLU(name="d_dense_1_activ"))


class EncoderTop(Network):

    def __init__(self):
        self.layers = []

        self.layers.append(Dense(128, name="e_dense_1"))
        self.layers.append(BatchNormalization(name="e_dense_bn_1", axis=-1))
        self.layers.append(LeakyReLU(name="e_dense_activ_1"))


class DiscriminatorTop(Network):

    def __init__(self):
        self.layers = []

        self.layers.append(Dense(1, name="d_classif_layer"))
        self.layers.append(Activation(activation=K.sigmoid, name="d_classif_activ"))
