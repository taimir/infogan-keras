import tensorflow as tf


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
