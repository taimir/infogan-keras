"""
Example implementation of InfoGAN
"""
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
from learn.models.infogan import InfoGAN2
from learn.models.infogan import InfoganDiscriminatorImpl, InfoganPriorImpl, \
    InfoganEncoderImpl, InfoganGeneratorImpl

# InfoganCheckpointer, InfoganTensorBoard, InfoganLogger
from learn.train.observers import InfoganLogger
from learn.train import ModelTrainer
from learn.data_management.skeleton_unsupervised import UnsupervisedSkeletonProvider
from learn.networks.rnns import RNNEncoderNetwork, RNNSharedNet, RNNDiscriminatorNetwork, \
    RNNGeneratorNetwork
from learn.stats.distributions import Categorical, IsotropicGaussian, IsotropicGaussian2


batch_size = 8

if __name__ == "__main__":
    # provide the data
    data_provider = UnsupervisedSkeletonProvider(
        data_path="/workspace/action_recogn/skeletons_data", batch_size=batch_size)
    val_x, val_y = data_provider.validation_data()

    recurrent_dim = val_x.shape[1]
    data_dim = val_x.shape[-1]

    meaningful_dists = {'c1': Categorical(n_classes=30),
                        'c2': IsotropicGaussian(dim=1),
                        'c3': IsotropicGaussian(dim=1)
                        }
    noise_dists = {'z': IsotropicGaussian(dim=42)}
    data_dist = IsotropicGaussian2(dim=1)
    prior_params = {'c1': {'p_vals': np.ones((batch_size, recurrent_dim, 30), dtype=np.float32) / 30},
                    'c2': {'mean': np.zeros((batch_size, recurrent_dim, 1), dtype=np.float32),
                           'std': np.ones((batch_size, recurrent_dim, 1), dtype=np.float32)},
                    'c3': {'mean': np.zeros((batch_size, recurrent_dim, 1), dtype=np.float32),
                           'std': np.ones((batch_size, recurrent_dim, 1), dtype=np.float32)},
                    'z': {'mean': np.zeros((batch_size, recurrent_dim, 42), dtype=np.float32),
                          'std': np.ones((batch_size, recurrent_dim, 42), dtype=np.float32)}
                    }

    prior = InfoganPriorImpl(meaningful_dists=meaningful_dists,
                             noise_dists=noise_dists,
                             prior_params=prior_params,
                             recurrent_dim=recurrent_dim)

    gen_net = RNNGeneratorNetwork(recurrent_dim=recurrent_dim,
                                  latent_dim=74,
                                  data_dim=data_dim,
                                  q_data_params_dim=2)

    generator = InfoganGeneratorImpl(data_shape=(data_dim, ),
                                     meaningful_dists=meaningful_dists,
                                     noise_dists=noise_dists,
                                     data_q_dist=data_dist,
                                     network=gen_net,
                                     recurrent_dim=recurrent_dim)

    shared_net = RNNSharedNet(recurrent_dim=recurrent_dim,
                              data_shape=(data_dim,))

    disc_net = RNNDiscriminatorNetwork(recurrent_dim=recurrent_dim,
                                       shared_out_shape=(32, ))
    discriminator = InfoganDiscriminatorImpl(network=disc_net)

    enc_net = RNNEncoderNetwork(recurrent_dim=recurrent_dim,
                                shared_out_shape=(32, ))
    encoder = InfoganEncoderImpl(batch_size=batch_size,
                                 meaningful_dists=meaningful_dists,
                                 supervised_dist=None,
                                 network=enc_net,
                                 recurrent_dim=recurrent_dim)

    model = InfoGAN2(batch_size=batch_size,
                     data_shape=(data_dim, ),
                     prior=prior,
                     generator=generator,
                     shared_net=shared_net,
                     discriminator=discriminator,
                     encoder=encoder,
                     recurrent_dim=recurrent_dim)

    from keras.utils import plot_model
    plot_model(model.gen_train_model, to_file='gen_train_model.png')
    plot_model(model.disc_train_model, to_file='disc_train_model.png')

    # define observers (callbacks during training)
    logger_observer = InfoganLogger(model=model, epoch_frequency=1)
    # tb_observer = InfoganTensorBoard(model=model, experiment_dir=sys.argv[1], epoch_frequency=1,
    # val_x=val_x, val_y=val_y)

    observers = [logger_observer]

    # train the model
    model_trainer = ModelTrainer(model, data_provider, observers)
    model_trainer.train(n_epochs=100)

    KTF.get_session().close()
