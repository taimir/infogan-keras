"""
Some network structures used by the InfoGAN
"""

import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Flatten, \
    Reshape
from keras.layers.advanced_activations import LeakyReLU


def generation_net(inputs, image_shape):
    network = inputs

    # a fully connected is needed to bring the inputs to a shape suitable for convolutions
    network = Dense(units=128, name="g_dense_1")(network)
    network = BatchNormalization(name="g_dense_bn_1")(network)
    network = Activation(activation=K.relu, name="g_dense_activ_1")(network)

    network = Dense(units=image_shape[1] // 4 * image_shape[2] // 4 * 32,
                    name="g_dense_2")(network)
    network = BatchNormalization(name="g_dense_bn_2")(network)
    network = Activation(activation=K.relu, name="g_dense_activ_2")(network)

    # I use the `th` orientation of theano
    network = Reshape(target_shape=(32, image_shape[1] // 4, image_shape[2] // 4),
                      name="g_reshape")(network)

    # start applying the deconv layers
    network = Conv2DTranspose(filters=16, kernel_size=(4, 4),
                              strides=(2, 2),
                              padding='same',
                              data_format='channels_first',
                              name="g_deconv_1")(network)
    network = BatchNormalization(name="g_deconv_bn_1")(network)
    network = Activation(activation=K.relu, name="g_deconv_activ_1")(network)

    # TODO: if we'll be generating color images, this needs to produce
    # a 256 * image_shape[0] number of channels
    network = Conv2DTranspose(filters=image_shape[0], kernel_size=(4, 4),
                              strides=(2, 2),
                              padding='same',
                              data_format='channels_first',
                              name="g_deconv_2")(network)
    network = Activation(activation=K.sigmoid, name="g_deconv_activ_2")(network)

    return network


def discrimination_net(inputs):
    network = inputs

    network = Conv2D(filters=8,
                     kernel_size=(5, 5),
                     padding="same",
                     name="d_conv_1")(network)
    network = LeakyReLU(name="d_conv_activ_1")(network)

    network = Conv2D(filters=16,
                     kernel_size=(3, 3),
                     padding="same",
                     name="d_conv_2")(network)
    network = BatchNormalization(name="d_conv_bn_2")(network)
    network = LeakyReLU(name="d_conv_activ_2")(network)

    network = Flatten(name="d_flatten")(network)
    network = Dense(units=128, name="d_dense_1")(network)
    network = BatchNormalization(name="d_dense_bn_1")(network)

    return network
