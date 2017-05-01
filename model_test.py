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
from sklearn.decomposition import PCA

from learn.models import InfoGAN
from learn.stats.distributions import Categorical, IsotropicGaussian, Bernoulli
from learn.utils.visualization import ROCView, micro_macro_roc, cluster_silhouette_view


batch_size = 256
n_classes = 10


def run_c1_only(roc_view, model, x_test, y_test, experiment_id):
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
    roc_view.add_curve(micro_fpr, micro_tpr, "infogan c1 only, micro")
    macro_fpr, macro_tpr = res['macro']
    roc_view.add_curve(macro_fpr, macro_tpr, "infogan c1 only, macro")


def run_svm(roc_view, model, x_train, y_train, x_test, y_test, experiment_id):
    # check the performance based on the original images with an SVM
    # training only on 5 % of the training data, simulating a semi-supervised scenario
    x_train = x_train.reshape((-1, 784))[:2500]
    y_train = y_train[:2500]
    x_test = x_test.reshape((-1, 784))

    classifier = svm.SVC()
    classifier.fit(x_train, y_train)

    test_preds = classifier.predict(x_test)
    acc = sum(y_test == test_preds) / len(y_test)
    print("Class. accuracy based on original representation: {}".format(acc))

    res = micro_macro_roc(n_classes,
                          y_expected=to_categorical(y_test, num_classes=n_classes),
                          y_predicted=to_categorical(test_preds, num_classes=n_classes))
    micro_fpr, micro_tpr = res['micro']
    roc_view.add_curve(micro_fpr, micro_tpr, "original SVM, micro")
    macro_fpr, macro_tpr = res['macro']
    roc_view.add_curve(macro_fpr, macro_tpr, "original SVM, macro")


def run_pca_svm(roc_view, model, x_train, y_train, x_test, y_test, experiment_id, n_pca=12):
    x_train = x_train.reshape((-1, 784))[:2500]
    y_train = y_train[:2500]
    x_test = x_test.reshape((-1, 784))

    pca = PCA(n_components=n_pca)

    pca.fit(x_train)

    # check the performance based on n_pca PCA features with an SVM
    test_encodings = pca.transform(x_test)
    train_encodings = pca.transform(x_train)

    classifier = svm.SVC()
    classifier.fit(train_encodings, y_train)

    test_preds = classifier.predict(test_encodings)
    acc = sum(y_test == test_preds) / len(y_test)
    print("Class. accuracy based on PCA latents: {}".format(acc))

    res = micro_macro_roc(n_classes,
                          y_expected=to_categorical(y_test, num_classes=n_classes),
                          y_predicted=to_categorical(test_preds, num_classes=n_classes))
    micro_fpr, micro_tpr = res['micro']
    roc_view.add_curve(micro_fpr, micro_tpr, "pca latent, micro")
    macro_fpr, macro_tpr = res['macro']
    roc_view.add_curve(macro_fpr, macro_tpr, "pca latent, macro")


def run_infogan_svm(roc_view, model, x_train, y_train, x_test, y_test, experiment_id):
    # check the performance based on all infogan features with an SVM
    test_encodings = np.concatenate(model.encode(x_test[:2500]), axis=1)
    y_test = y_test[:2500]
    train_encodings = np.concatenate(model.encode(x_train), axis=1)

    classifier = svm.SVC()
    classifier.fit(train_encodings, y_train)

    test_preds = classifier.predict(test_encodings)
    acc = sum(y_test == test_preds) / len(y_test)
    print("Class. accuracy based on InfoGAN latents: {}".format(acc))

    res = micro_macro_roc(n_classes,
                          y_expected=to_categorical(y_test, num_classes=n_classes),
                          y_predicted=to_categorical(test_preds, num_classes=n_classes))
    micro_fpr, micro_tpr = res['micro']
    roc_view.add_curve(micro_fpr, micro_tpr, "infogan latent, micro")
    macro_fpr, macro_tpr = res['macro']
    roc_view.add_curve(macro_fpr, macro_tpr, "infogan latent, macro")


def run_cluster_evaluation(model, x_test, y_test, experiment_id):
    test_encodings = np.concatenate(model.encode(x_test), axis=1)
    # produce a clustering evaluation
    cluster_silhouette_view(test_encodings, y_test,
                            os.path.join(experiment_id, "silhouette_score.png"),
                            n_clusters=n_classes)


def test_mnist_performance(model, x_test, y_test, x_train, y_train, experiment_id):
    roc_view = ROCView()

    run_svm(roc_view, model, x_train, y_train, x_test, y_test, experiment_id)
    run_pca_svm(roc_view, model, x_train, y_train, x_test, y_test, experiment_id)
    run_infogan_svm(roc_view, model, x_train, y_train, x_test, y_test, experiment_id)

    run_cluster_evaluation(model, x_test, y_test, experiment_id)

    roc_view.save_and_close(os.path.join(experiment_id, "ROC.png"))


if __name__ == "__main__":
    experiment_id = sys.argv[1]

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
                    supervised_dist_name="c1")

    gen_weights_filepath = os.path.join(experiment_id, "gen_train_model.hdf5")
    disc_weights_filepath = os.path.join(experiment_id, "disc_train_model.hdf5")
    model.load_weights(gen_weights_filepath, disc_weights_filepath)

    test_mnist_performance(model, x_test, y_test, x_train, y_train, experiment_id)
    KTF.get_session().close()
