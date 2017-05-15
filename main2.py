"""
Example implementation of InfoGAN
"""
# import sys
# import os
# import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# def get_session(gpu_fraction=0.8):
# num_threads = os.environ.get('OMP_NUM_THREADS')
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
# allow_growth=True)

# if num_threads:
# return tf.Session(config=tf.ConfigProto(
# gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
# else:
# return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# KTF.set_session(get_session())

import numpy as np
from learn.models.infogan_advanced import InfoGAN2
from learn.models.infogan_advanced import InfoganDiscriminatorImpl, InfoganPriorImpl, \
    InfoganEncoderImpl, InfoganGeneratorImpl

# InfoganCheckpointer, InfoganTensorBoard, InfoganLogger
from learn.train.observers import InfoganLogger
from learn.train import ModelTrainer
from learn.data_management import SemiSupervisedMNISTProvider

from learn.stats.distributions import Categorical, IsotropicGaussian, Bernoulli


batch_size = 1

if __name__ == "__main__":
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

    prior = InfoganPriorImpl(shape_prefix=(),
                             meaningful_dists=meaningful_dists,
                             noise_dists=noise_dists,
                             prior_params=prior_params)

    generator = InfoganGeneratorImpl(shape_prefix=(),
                                     data_shape=(28, 28, 1),
                                     meaningful_dists=meaningful_dists,
                                     noise_dists=noise_dists,
                                     data_q_dist=image_dist,
                                     network=)

    discriminator = InfoganDiscriminatorImpl(network=)

    encoder = InfoganEncoderImpl(recurrent=False,
                                 meaningful_dists=meaningful_dists,
                                 supervised_dist=None,
                                 network=)
    model = InfoGAN2(prior=prior,
                     generator=generator,
                     discriminator=discriminator,
                     encoder=encoder)

    from keras.utils import plot_model
    plot_model(model.gen_train_model, to_file='gen_train_model.png')
    plot_model(model.disc_train_model, to_file='disc_train_model.png')

    # provide the data
    data_provider = SemiSupervisedMNISTProvider(batch_size)
    val_x, val_y = data_provider.validation_data()

    # define observers (callbacks during training)
    logger_observer = InfoganLogger(model=model, epoch_frequency=1)

    observers = [logger_observer]

    # train the model
    model_trainer = ModelTrainer(model, data_provider, observers)
    model_trainer.train(n_epochs=100)

    KTF.get_session().close()
