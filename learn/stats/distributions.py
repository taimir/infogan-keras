"""
Module with simple distribution definitions around keras.
Mainly needed because PDF functions are not defines in keras.
"""

import abc

import keras.backend as K
from keras.activations import linear, softmax, softplus, sigmoid
import numpy as np


def cholesky(square_mat):
    """
    cholesky perfoms a cholesky decomposition on a keras variable

    :param square_mat - a square positive definite matrix
    """

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        L = tf.cholesky(square_mat)
        return L
    else:
        import theano.tensor.slinalg as alg
        L = alg.cholesky(square_mat)
        return L


class Distribution(object):
    """
    Abstract distribution class
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample(self, param_dict):
        raise NotImplementedError

    @abc.abstractmethod
    def nll(self, samples, param_dict):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def param_info(self):
        raise NotImplementedError


class IsotropicGaussian(Distribution):

    def __init__(self, dim):
        self.dim = dim

    def sample(self, param_dict):
        mean = param_dict['mean']
        return K.random_uniform(shape=K.shape(mean), minval=-1.0, maxval=+1.0)

    def sample2(self, param_dict):
        mean = param_dict['mean']
        std = param_dict['std']
        eps = K.random_normal(shape=K.shape(mean), mean=0, stddev=1.)
        sample = mean + std * eps
        return sample

    def nll(self, samples, param_dict, use_std=False):
        mean = param_dict['mean']
        # using the std. dev should be done with caution, as it can
        # result in very large gradients in the beginning of training
        if use_std:
            std = param_dict['std']
        else:
            std = 1.0

        return K.sum(
            0.5 * np.log(2 * np.pi) + K.log(std + K.epsilon()) +
            0.5 * K.square((samples - mean) / (std + K.epsilon())),
            axis=-1)

    def sample_size(self):
        return self.dim

    def param_info(self):
        # since we are isotropic, only one number for the std. dev
        return {
            'mean': (self.dim, linear),
            'std': (self.dim, softplus)
        }


class Categorical(Distribution):

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def sample(self, param_dict):
        p_vals = param_dict['p_vals']
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            shape = tf.shape(p_vals)
            reshaped_pvals = tf.reshape(p_vals, [-1, self.n_classes])
            samples = tf.multinomial(logits=tf.log(reshaped_pvals), num_samples=1)[:, 0]
            # a hack to turn it into one-hot
            onehot = tf.constant(np.eye(self.n_classes, dtype=np.float32))
            result = tf.nn.embedding_lookup(onehot, samples)
            return tf.reshape(result, shape)
        else:
            from theano.tensor.shared_randomstreams import RandomStreams
            random = RandomStreams()
            return random.multinomial(size=K.shape(p_vals)[:-1], n=1, pvals=p_vals,
                                      dtype='float32')

    def nll(self, samples, param_dict):
        """log_pdf

        :param param_dict - { 'p_vals': ...}
        """
        p_vals = param_dict['p_vals']

        return -K.sum(samples * K.log(p_vals + K.epsilon()), axis=-1)

    def sample_size(self):
        return self.n_classes

    def param_info(self):
        return {
            'p_vals': (self.n_classes, softmax)
        }


class Bernoulli(Distribution):

    def sample(self, param_dict):
        p = param_dict['p']
        # return K.random_binomial(shape=K.shape(p), p=p)

        # TODO: for now, just return the mean
        # but this needs to be fixed with the SGVB reparametrization
        return p

    def nll(self, samples, param_dict):
        """log_pdf

        :param param_dict - { 'p_vals': ...}
        """
        p_vals = param_dict['p']
        return K.mean(-K.sum(
            samples * K.log(p_vals + K.epsilon()) + (1 - samples) * K.log(1 - p_vals + K.epsilon()),
            axis=1))

    def sample_size(self):
        return 1

    def param_info(self):
        return {
            'p': (1, sigmoid)
        }
