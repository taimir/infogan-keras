"""
Example implementation of InfoGAN
"""
import sys
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def get_session(gpu_fraction=0.8):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from learn.models.infogan import InfoGAN
from learn.models.model_trainer import ModelTrainer
from learn.stats.distributions import Categorical, IsotropicGaussian, Bernoulli


batch_size = 256

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1)) / 255
    x_test = x_test.reshape((-1, 28, 28, 1)) / 255

    x_val = x_train[:1000]
    y_val = y_train[:1000]
    x_train = x_train[1000:]

    datagen = ImageDataGenerator(data_format='channels_last')
    datagen.fit(x_train)

    def data_generator():
        return datagen.flow(x_train, batch_size=batch_size)

    meaningful_dists = {'c1': Categorical(n_classes=10),
                        'c2': IsotropicGaussian(dim=1),
                        'c3': IsotropicGaussian(dim=1)
                        }
    noise_dists = {'z': IsotropicGaussian(dim=62)}
    image_dist = Bernoulli()
    prior_params = {'c1': {'p_vals': np.ones((batch_size, 10), dtype=np.float32) / 10},
                    'c2': {'mean': np.zeros((batch_size, 1), dtype=np.float32),
                           'std': np.ones((batch_size, 1), dtype=np.float32)},
                    'c3': {'mean': np.zeros((batch_size, 1), dtype=np.float32),
                           'std': np.ones((batch_size, 1), dtype=np.float32)},
                    'z': {'mean': np.zeros((batch_size, 62), dtype=np.float32),
                          'std': np.ones((batch_size, 62), dtype=np.float32)}
                    }

    model = InfoGAN(batch_size=batch_size,
                    image_shape=(28, 28, 1),
                    noise_dists=noise_dists,
                    meaningful_dists=meaningful_dists,
                    image_dist=image_dist,
                    prior_params=prior_params,
                    experiment_id=sys.argv[1])

    plot_model(model.gen_train_model, to_file='gen_train_model.png')
    plot_model(model.disc_train_model, to_file='disc_train_model.png')
    plot_model(model.encoder_model, to_file='encoder_model.png')
    plot_model(model.disc_model, to_file='disc_model.png')
    plot_model(model.gen_model, to_file='gen_model.png')

    model_trainer = ModelTrainer(model, data_generator, x_val, y_val)

    model_trainer.train()
