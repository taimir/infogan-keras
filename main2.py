"""
Example implementation of InfoGAN
"""
import sys
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_session(gpu_fraction=0.2):
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
from learn.models.infogan_advanced import InfoGAN2
from learn.models.infogan_advanced import InfoganDiscriminatorImpl, InfoganPriorImpl, \
    InfoganEncoderImpl, InfoganGeneratorImpl

# InfoganCheckpointer, InfoganTensorBoard, InfoganLogger
from learn.train.observers import InfoganLogger, InfoganTensorBoard
from learn.train import ModelTrainer
from learn.data_management import SemiSupervisedMNISTProvider
from learn.networks.convnets import EncoderNetwork, SharedNet, DiscriminatorNetwork, \
    BinaryImgGeneratorNetwork
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

    prior = InfoganPriorImpl(meaningful_dists=meaningful_dists,
                             noise_dists=noise_dists,
                             prior_params=prior_params,
                             recurrent_dim=None)

    gen_net = BinaryImgGeneratorNetwork(image_shape=(28, 28, 1))
    generator = InfoganGeneratorImpl(data_param_shape=(28, 28, 1),
                                     data_shape=(28, 28, 1),
                                     meaningful_dists=meaningful_dists,
                                     noise_dists=noise_dists,
                                     data_q_dist=image_dist,
                                     network=gen_net,
                                     recurrent_dim=None)

    shared_net = SharedNet()

    disc_net = DiscriminatorNetwork(shared_net=shared_net)
    discriminator = InfoganDiscriminatorImpl(network=disc_net)

    enc_net = EncoderNetwork(shared_net=shared_net)
    encoder = InfoganEncoderImpl(batch_size=batch_size,
                                 meaningful_dists=meaningful_dists,
                                 supervised_dist=None,
                                 network=enc_net,
                                 recurrent_dim=None)

    model = InfoGAN2(batch_size=batch_size,
                     data_shape=(28, 28, 1),
                     prior=prior,
                     generator=generator,
                     discriminator=discriminator,
                     encoder=encoder,
                     recurrent_dim=None)

    from keras.utils import plot_model
    plot_model(model.gen_train_model, to_file='gen_train_model.png')
    plot_model(model.disc_train_model, to_file='disc_train_model.png')

    # provide the data
    data_provider = SemiSupervisedMNISTProvider(batch_size)
    val_x, val_y = data_provider.validation_data()

    # define observers (callbacks during training)
    logger_observer = InfoganLogger(model=model, epoch_frequency=1)
    tb_observer = InfoganTensorBoard(model=model, experiment_dir=sys.argv[1], epoch_frequency=1,
                                     val_x=val_x, val_y=val_y)

    observers = [logger_observer, tb_observer]

    # train the model
    model_trainer = ModelTrainer(model, data_provider, observers)
    model_trainer.train(n_epochs=100)

    KTF.get_session().close()
