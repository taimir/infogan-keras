"""
Example implementation of InfoGAN
"""

import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

from learn.models.infogan import InfoGAN
from learn.models.model_trainer import ModelTrainer
from learn.stats.distributions import Categorical, IsotropicGaussian


batch_size = 32

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    def data_generator():
        return datagen.flow(x_train, batch_size=batch_size)

    meaningful_dists = {'c1': Categorical(batch_size=batch_size, n_classes=10),
                        'c2': IsotropicGaussian(batch_size=batch_size, dim=1),
                        'c3': IsotropicGaussian(batch_size=batch_size, dim=1)}
    noise_dists = {'z': IsotropicGaussian(batch_size=batch_size, dim=30)}
    image_dist = None
    prior_params = {'c1': {'p_vals': np.ones((batch_size, 10), dtype=np.float32) / 10},
                    'c2': {'mean': np.zeros((batch_size, 1)), 'std': 1.0},
                    'c3': {'mean': np.zeros((batch_size, 1)), 'std': 1.0},
                    'z': {'mean': np.zeros((batch_size, 30)), 'std': 1.0}}

    model = InfoGAN(batch_size=batch_size,
                    image_shape=(1, 28, 28),
                    noise_dists=noise_dists,
                    meaningful_dists=meaningful_dists,
                    image_dist=image_dist,
                    prior_params=prior_params)

    model_trainer = ModelTrainer(model, data_generator)

    model_trainer.train()
