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
from scipy.stats import mode
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn import svm

from learn.models.infogan import InfoGAN
from learn.stats.distributions import Categorical, IsotropicGaussian, Bernoulli
from learn.utils.visualization import ROCView, micro_macro_roc


batch_size = 256


def test_mnist_performance(model, x_test, y_test, x_train, y_train):
    roc_view = ROCView()
    n_classes = 10

    # encodings: list of arrays with shape (N, salient_dim)
    encodings_list = model.encode(x_test)

    # check performance only based on c1 classification
    c1 = [e for e in encodings_list if e.shape[1] == 10][0]
    c1 = np.argmax(c1, axis=1)

    # TODO: if the model is not too god, in some cases the class
    # coverage might not be complete
    c1_map = np.zeros_like(y_test)
    for digit in range(10):
        digit_map = mode(c1[y_test == digit])[0][0]
        c1_map[c1 == digit_map] = digit

    acc = sum(y_test == c1_map) / len(y_test)
    print("Class. accuracy based on c1 (categorical latent): {}".format(acc))

    res = micro_macro_roc(n_classes,
                          y_expected=to_categorical(y_test, num_classes=n_classes),
                          y_predicted=to_categorical(c1_map, num_classes=n_classes))
    micro_fpr, micro_tpr = res['micro']
    roc_view.add_curve(micro_fpr, micro_tpr, "categorical only, micro")
    macro_fpr, macro_tpr = res['macro']
    roc_view.add_curve(macro_fpr, macro_tpr, "categorical only, macro")

    # check the performance based on all features
    # with an SVM
    test_encodings = np.concatenate(encodings_list, axis=1)
    train_encodings = np.concatenate(model.encode(x_train), axis=1)

    classifier = svm.SVC()
    classifier.fit(train_encodings, y_train)

    test_preds = classifier.predict(test_encodings)
    acc = sum(y_test == test_preds) / len(y_test)
    print("Class. accuracy based on all latents: {}".format(acc))

    res = micro_macro_roc(n_classes,
                          y_expected=to_categorical(y_test, num_classes=n_classes),
                          y_predicted=to_categorical(test_preds, num_classes=n_classes))
    micro_fpr, micro_tpr = res['micro']
    roc_view.add_curve(micro_fpr, micro_tpr, "all latent, micro")
    macro_fpr, macro_tpr = res['macro']
    roc_view.add_curve(macro_fpr, macro_tpr, "all latent, macro")

    roc_view.save_and_close("ROC.png")


if __name__ == "__main__":
    gen_weights_filepath = sys.argv[1]
    disc_weights_filepath = sys.argv[2]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1)) / 255
    x_test = x_test.reshape((-1, 28, 28, 1)) / 255

    x_val = x_train[:1000]
    y_val = y_train[:1000]
    x_train = x_train[1000:]
    y_train = y_train[1000:]

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
                    experiment_id="testing")

    model.load_weights(gen_weights_filepath, disc_weights_filepath)

    test_mnist_performance(model, x_test, y_test, x_train, y_train)
