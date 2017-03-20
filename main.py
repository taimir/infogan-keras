"""
Example implementation of InfoGAN
"""

import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from learn.models.infogan import InfoGAN
from learn.models.model_trainer import ModelTrainer
from learn.stats.distributions import Categorical, IsotropicGaussian, Bernoulli


batch_size = 2

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 1, 28, 28)) / 255
    x_test = x_test.reshape((-1, 1, 28, 28)) / 255

    datagen = ImageDataGenerator(data_format='channels_first')
    datagen.fit(x_train)

    def data_generator():
        return datagen.flow(x_train, batch_size=batch_size)

    meaningful_dists = {'c1': Categorical(n_classes=10),
                        'c2': IsotropicGaussian(dim=1),
                        'c3': IsotropicGaussian(dim=1)}
    noise_dists = {'z': IsotropicGaussian(dim=30)}
    image_dist = Bernoulli()
    prior_params = {'c1': {'p_vals': np.ones((batch_size, 10), dtype=np.float32) / 10},
                    'c2': {'mean': np.zeros((batch_size, 1), dtype=np.float32),
                           'std': np.ones((batch_size, 1), dtype=np.float32)},
                    'c3': {'mean': np.zeros((batch_size, 1), dtype=np.float32),
                           'std': np.ones((batch_size, 1), dtype=np.float32)},
                    'z': {'mean': np.zeros((batch_size, 30), dtype=np.float32),
                          'std': np.ones((batch_size, 30), dtype=np.float32)}
                    }

    model = InfoGAN(batch_size=batch_size,
                    image_shape=(1, 28, 28),
                    noise_dists=noise_dists,
                    meaningful_dists=meaningful_dists,
                    image_dist=image_dist,
                    prior_params=prior_params)

    plot_model(model.enc_gen_model, to_file='gan_model.png')
    plot_model(model.disc_model, to_file='disc_model.png')
    plot_model(model.encoder_model, to_file='encoder_model.png')

    model_trainer = ModelTrainer(model, data_generator)

    model_trainer.train()
