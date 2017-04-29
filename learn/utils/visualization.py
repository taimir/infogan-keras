
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import tensorflow as tf
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve, silhouette_score, silhouette_samples

colors = ['#991012', '#c4884e', '#93bf8d', '#a3dbff']
sns.set_palette(colors)


def image_grid(input_tensor, grid_shape, image_shape):
    """
    form_image_grid forms a grid of image tiles from input_tensor.

    :param input_tensor - batch of images, shape (N, height, width, n_channels)
    :param grid_shape - shape (in tiles) of the grid, e.g. (10, 10)
    :param image_shape - shape of a single image, e.g. (28, 28, 1)
    """
    # take the subset of images
    input_tensor = input_tensor[:grid_shape[0] * grid_shape[1]]
    # add black tiles if needed
    required_pad = grid_shape[0] * grid_shape[1] - tf.shape(input_tensor)[0]

    def add_pad():
        padding = tf.zeros((required_pad,) + image_shape)
        return tf.concat([input_tensor, padding], axis=0)
    input_tensor = tf.cond(required_pad > 0, add_pad, lambda: input_tensor)

    # height and width of the grid
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    # form the grid tensor
    input_tensor = tf.reshape(input_tensor, grid_shape + image_shape)
    # flip height and width
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    # form the rows
    input_tensor = tf.reshape(input_tensor, [grid_shape[0], width, image_shape[0], image_shape[2]])
    # flip width and height again
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    # form the columns
    input_tensor = tf.reshape(input_tensor, [1, height, width, image_shape[2]])
    return input_tensor


class ROCView(object):
    """
    ROCView generates and plots the ROC curves of a model.
    The view is created in a way that allows multiple ROC curves to be added to it before
    it is saved.

    Usage:
        >>> tpr = [0.3, 1, 1]
        >>> fpr = [0, 0.4, 1]
        >>> view = ROCView("my/data/dir")
        >>> view.add_curve(fpr=fpr, tpr=tpr, label="ROC of model 1")
        >>> # you can call view.add_curve() again if needed
        >>> view.save_and_close("example_file.png")
    """

    def __init__(self):
        self.ax, self.fig = self._init_ROC()

    def _init_ROC(self):
        """
        initialise the plots (figure, axes)
        :return:
        """
        sns.set_style("whitegrid")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_aspect(1)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.axes().set_aspect('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False positive rate', size=10)
        plt.ylabel('True positive rate', size=10)
        plt.title('Receiver operating characteristic', size=15)

        return ax, fig

    def add_curve(self, fpr, tpr, label):
        """
        computes and draws a ROC curve for the given TPR and FPR, adds a legend with the specified
        label and the AUC score

        :param fpr: array, false positive rate
        :param tpr: array, true positive rate
        :param label: text to be put into the legend entry for this curve
        """
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='{0} (AUC = {1:0.2f})'.format(label, roc_auc))

    def save_and_close(self, file_path):
        """
        saves the figure into a file.

        :param file_path: path to the file for the figure of the ROC curve
        :return:
        """
        # Put a legend below current axis
        self.ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=1, prop={'size': 9},
                       frameon=True)
        self.fig.savefig(filename=file_path, bbox_inches='tight')


def micro_macro_roc(n_classes, y_expected, y_predicted):
    """
    MicroMacroROC can create the TPR (True positive rate) and FPR (false positive rate)
    for two different ROC curves based on multi-class classification results:
    * "micro" : fpr, tpr are computed for the flattened predictions for all
    classes (i.e. all predictions combined). Weakly represented classes
    thus contribute less to the "micro" curve.
    * "macro" : fpr, tpr are computed as an average of the ROC
    curves for each of the classes. Thus every class is treated as
    equally important in the "macro" curve.

    :param n_classes: how many classes does the classifier predict for
    :param y_expected: a numpy array of expected class labels
    (1-hot encoded)
    :param y_predicted: a numpy array of prediction scores
    :return: {
                "micro": (fpr, tpr),
                "macro": (fpr, tpr)
            }
    """

    # Compute micro-average ROC curve
    micro_fpr, micro_tpr, _ = roc_curve(y_expected.ravel(), y_predicted.ravel())

    # Compute macro-average ROC curve
    # First aggregate all false positive rates per class into one array
    per_class_fpr = dict()
    per_class_tpr = dict()
    for i in range(n_classes):
        per_class_fpr[i], per_class_tpr[i], _ = roc_curve(y_expected[:, i], y_predicted[:, i])
    all_fpr = np.unique(np.concatenate([per_class_fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, per_class_fpr[i], per_class_tpr[i])

    # Finally average it
    mean_tpr /= float(n_classes)

    macro_fpr = all_fpr
    macro_tpr = mean_tpr

    return {
        "micro": (micro_fpr, micro_tpr),
        "macro": (macro_fpr, macro_tpr)
    }


def cluster_silhouette_view(X, y, file_path, n_clusters):
    # initialize the figure
    sns.set_style("whitegrid")

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.xlim([-0.5, 1.0])
    plt.ylim([0, X.shape[0] + (n_clusters + 1) * 10])
    plt.xlabel('Silhouette score per sample', size=10)
    plt.ylabel('Samples in clusters', size=10)
    plt.title('Silhouette scores', size=15)

    # compute the silhoette average score of the clustering
    score_avg = silhouette_score(X, y)
    print("The average silhouette score is :", score_avg)

    # Compute the silhouette scores for each sample
    score_per_sample = silhouette_samples(X, y)

    y_lower = 10
    for i in range(n_clusters):
        # scores of the samples in i'th cluster, sorted
        score_per_sample_i = score_per_sample[y == i]
        score_per_sample_i.sort()

        size_cluster_i = score_per_sample_i.shape[0]

        # do the plotting of the diagram
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, score_per_sample_i, alpha=0.7,
                          facecolor=color,
                          edgecolor=color,
                          label="cluster {}".format(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=score_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=1, prop={'size': 9},
              frameon=True)
    fig.savefig(filename=file_path, bbox_inches='tight')
