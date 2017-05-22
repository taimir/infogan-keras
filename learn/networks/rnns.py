"""
Recurrent network structures used by the InfoGAN
"""
import keras.backend as K
from keras.layers import BatchNormalization, Activation, Dense, Reshape, GRU, TimeDistributed, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from learn.networks.interfaces import Network


class RNNGeneratorNetwork(Network):

    def __init__(self, recurrent_dim, latent_dim, data_dim, q_data_params_dim):
        self.layers = []

        self.layers.append(GRU(64, activation="relu", return_sequences=True))

        self.layers.append(TimeDistributed(Dense(units=64, name="g_dense_1")))
        # self.layers.append(TimeDistributed(BatchNormalization(name="g_dense_bn_1", axis=-1)))
        self.layers.append(TimeDistributed(Activation(activation=K.relu, name="g_dense_activ_1")))

        self.layers.append(TimeDistributed(Dense(units=64,
                                                 name="g_dense_2")))
        # self.layers.append(TimeDistributed(BatchNormalization(name="g_dense_bn_2", axis=-1)))
        self.layers.append(TimeDistributed(Activation(activation=K.relu, name="g_dense_activ_2")))

        self.layers.append(TimeDistributed(Dense(units=128,
                                                 name="g_dense_3")))
        # self.layers.append(TimeDistributed(BatchNormalization(name="g_dense_bn_3", axis=-1)))
        self.layers.append(TimeDistributed(Activation(activation=K.relu, name="g_dense_activ_3")))

        self.layers.append(TimeDistributed(Dense(units=data_dim * q_data_params_dim,
                                                 name="g_dense_3")))

        self.layers.append(Reshape(target_shape=(recurrent_dim, q_data_params_dim, data_dim),
                                   name="g_param_reshape"))

        inputs = Input(shape=(recurrent_dim, latent_dim))
        network = inputs
        for layer in self.layers:
            network = layer(network)

        self.model = Model(inputs=[inputs], outputs=[network], name="G")

    def apply(self, inputs):
        return self.model(inputs)


class RNNSharedNet(Network):
    """
    RNNSharedNet

    The convolutional structure is applied to the last output of the RNN.
    """

    def __init__(self, recurrent_dim, data_shape):
        self.layers = []

        self.layers.append(GRU(64, activation="relu", return_sequences=True))

        self.layers.append(TimeDistributed(Dense(256)))
        self.layers.append(TimeDistributed(LeakyReLU(name="d_conv_activ_1")))

        self.layers.append(TimeDistributed(Dense(128)))
        # self.layers.append(TimeDistributed(BatchNormalization(name="d_conv_bn_2", axis=-1)))
        self.layers.append(TimeDistributed(LeakyReLU(name="d_conv_activ_2")))

        self.layers.append(TimeDistributed(Dense(128)))
        # self.layers.append(TimeDistributed(BatchNormalization(name="d_conv_bn_3", axis=-1)))
        self.layers.append(TimeDistributed(LeakyReLU(name="d_conv_activ_3")))

        self.layers.append(TimeDistributed(Dense(64)))
        # self.layers.append(TimeDistributed(BatchNormalization(name="d_conv_bn_4", axis=-1)))
        self.layers.append(TimeDistributed(LeakyReLU(name="d_conv_activ_4")))

        self.layers.append(TimeDistributed(Dense(32, activation="relu")))
        # self.layers.append(TimeDistributed(BatchNormalization(name="d_dense_bn_1", axis=-1)))
        self.layers.append(TimeDistributed(LeakyReLU(name="d_dense_1_activ")))

        inputs = Input(shape=(recurrent_dim,) + data_shape)
        network = inputs
        for layer in self.layers:
            network = layer(network)

        self.model = Model(inputs=[inputs], outputs=[network], name="SHARED")

    def apply(self, inputs):
        return self.model(inputs)


class RNNEncoderNetwork(Network):

    def __init__(self, recurrent_dim, shared_out_shape):
        self.layers = []
        self.layers.append(TimeDistributed(Dense(32, name="e_dense_1")))
        # self.layers.append(TimeDistributed(BatchNormalization(name="e_dense_bn_1",
        # axis=-1, scale=False)))
        self.layers.append(TimeDistributed(LeakyReLU(name="e_dense_activ_1")))

        inputs = Input(shape=(recurrent_dim,) + shared_out_shape)
        network = inputs
        for layer in self.layers:
            network = layer(network)

        self.model = Model(inputs=[inputs], outputs=[network], name="E_top")

    def apply(self, inputs):
        return self.model(inputs)


class RNNDiscriminatorNetwork(Network):

    def __init__(self, recurrent_dim, shared_out_shape):
        self.layers = []
        self.layers.append(TimeDistributed(Dense(1, name="d_classif_layer")))

        inputs = Input(shape=(recurrent_dim,) + shared_out_shape)
        network = inputs
        for layer in self.layers:
            network = layer(network)

        self.model = Model(inputs=[inputs], outputs=[network], name="D_top")

    def apply(self, inputs):
        return self.model(inputs)
