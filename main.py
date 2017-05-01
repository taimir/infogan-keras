"""
Example implementation of InfoGAN
"""
import sys
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
from learn.models import InfoGAN
from learn.train.observers import InfoganCheckpointer, InfoganTensorBoard, InfoganLogger
from learn.train import ModelTrainer
from learn.data_management import SemiSupervisedMNISTProvider

from learn.stats.distributions import Categorical, IsotropicGaussian, Bernoulli


batch_size = 256

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

    model = InfoGAN(batch_size=batch_size,
                    image_shape=(28, 28, 1),
                    noise_dists=noise_dists,
                    meaningful_dists=meaningful_dists,
                    image_dist=image_dist,
                    prior_params=prior_params,
                    supervised_dist_name="c1")

    # from keras.utils import plot_model
    # plot_model(model.gen_train_model, to_file='gen_train_model.png')
    # plot_model(model.disc_train_model, to_file='disc_train_model.png')
    # plot_model(model.encoder_model, to_file='encoder_model.png')
    # plot_model(model.disc_model, to_file='disc_model.png')
    # plot_model(model.gen_model, to_file='gen_model.png')

    # provide the data
    data_provider = SemiSupervisedMNISTProvider(batch_size)
    val_x, val_y = data_provider.validation_data()

    # define observers (callbacks during training)
    tb_observer = InfoganTensorBoard(model=model, experiment_dir=sys.argv[1], epoch_frequency=1,
                                     val_x=val_x, val_y=val_y)
    checkpoint_observer = InfoganCheckpointer(model=model, experiment_dir=sys.argv[1],
                                              epoch_frequency=5)
    logger_observer = InfoganLogger(model=model, epoch_frequency=1)

    observers = [tb_observer, checkpoint_observer, logger_observer]

    # train the model
    model_trainer = ModelTrainer(model, data_provider, observers)
    model_trainer.train(n_epochs=100)

    KTF.get_session().close()
