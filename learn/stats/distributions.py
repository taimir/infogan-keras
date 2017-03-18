"""
Module with simple distribution definitions around keras.
Mainly needed because PDF functions are not defines in keras.
"""

import abc

import keras.backend as K
from keras.activations import linear, softmax, softplus
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


class Distribution(abc.ABCMeta):
    """
    Abstract distribution class
    """

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

    def __init__(self, batch_size, dim):
        self.batch_size = batch_size
        self.dim = dim

    def sample(self, param_dict):
        mean = param_dict['mean']
        std = param_dict['std']
        eps = K.random_normal(shape=K.shape(mean), mean=0, std=1.)
        sample = mean + std * eps
        return sample

    def nll(self, samples, param_dict):
        mean = param_dict['mean']
        std = param_dict['std']
        return K.mean(-K.sum(
            -0.5 * np.log(2 * np.pi) - K.log(std) - 0.5 * K.square((samples - mean) / std),
            axis=-1))

    def sample_size(self):
        return self.dim

    def param_info(self):
        # since we are isotropic, only one number for the std. dev
        return {
            'mean': (self.dim, linear),
            'std': (1, softplus)
        }


class Categorical(Distribution):

    def __init__(self, batch_size, n_classes):
        self.batch_size = batch_size
        self.n_classes = n_classes

    def sample(self, param_dict):
        p_vals = param_dict['p_vals']
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            samples = tf.multinomial(logits=tf.log(p_vals), num_samples=1)[:, 0]
            # a hack to turn it into one-hot
            onehot = tf.constant(np.eye(self.n_classes, dtype=np.float32))
            return tf.nn.embedding_lookup(onehot, samples)
        else:
            from theano.tensor.shared_randomstreams import RandomStreams
            random = RandomStreams()
            return random.multinomial(size=K.shape(p_vals), n=1, pvals=p_vals)

    def nll(self, samples, param_dict):
        """log_pdf

        :param samples - one-hot encoded categorical samples, batch_size many
        :param param_dict - { 'p_vals': ...}
        """
        p_vals = param_dict['p_vals']
        return K.mean(-K.sum(
            samples * K.log(p_vals),
            axis=1))

    def sample_size(self):
        return self.n_classes

    def param_info(self):
        return {
            'p_vals': (self.n_classes, softmax)
        }


class Bernoulli(Distribution):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self, param_dict):
        p = param_dict['p']
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            shape = K.shape(p)
            return tf.select(tf.random_uniform(shape=K.shape(p)) - p > 0.0,
                             tf.ones(shape),
                             tf.zeros(shape))
        else:
            from theano.tensor.shared_randomstreams import RandomStreams
            random = RandomStreams()
            return random.binomial(size=K.shape(p), n=1, p=p)

    def nll(self, samples, param_dict):
        """log_pdf

        :param samples - one-hot encoded categorical samples, batch_size many
        :param param_dict - { 'p_vals': ...}
        """
        p_vals = param_dict['p_vals']
        return K.mean(-K.sum(
            samples * K.log(p_vals) + (1 - samples) * K.log(1 - p_vals),
            axis=1))

    def sample_size(self):
        return 1

    def param_info(self):
        return {
            'p_vals': (self.n_classes, softmax)
        }
