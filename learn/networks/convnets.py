"""
Some network structures used by the InfoGAN
"""

import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Deconv2D
from keras.layers.advanced_activations import LeakyReLU

def generation_net(inputs, image_shape):
    network = inputs

    # a fully connected is needed to bring the inputs to a shape suitable for convolutions
    network = Dense(1024)(network)
    network = BatchNormalization()(network)
    network = Activation(activation=K.relu)(network)

    network = Dense(image_shape[1] // 4 * image_shape[2] // 4 * 128)(network)
    network = BatchNormalization()(network)
    network = Activation(activation=K.relu)(network)

    # I use the `th` orientation of theano
    network = K.reshape(network, shape=(-1, 128, image_shape[1] // 4, image_shape[2] // 4))

    # start applying the deconv layers
    output_shape = (None, 64, image_shape[1] // 2, image_shape[2] // 2)
    network = Deconv2D(nb_filter=64, nb_row=4, nb_col=4, output_shape=output_shape)(network)
    network = BatchNormalization()(network)
    network = Activation(activation=K.relu)(network)

    # TODO: if we'll be generating color images, this needs to produce
    # a 256 * image_shape[0] number of channels
    output_shape = (None,) + image_shape
    network = Deconv2D(nb_filter=image_shape[0], nb_row=4, nb_col=4,
                       output_shape=output_shape)(network)
    network = Activation(activation=K.sigmoid)(network)

    return network


def discrimination_net(inputs):
    network = inputs

    network = Conv2D(nb_filter=64, nb_row=4, nb_col=4)(network)
    network = LeakyReLU()(network)

    network = Conv2D(nb_filter=128, nb_row=4, nb_col=4)(network)
    network = BatchNormalization()(network)
    network = LeakyReLU()(network)

    network = Dense(output_dim=1024)(network)
    network = BatchNormalization(network)

    return network

