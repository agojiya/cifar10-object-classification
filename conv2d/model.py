import tensorflow as tf
from fileio import cifar10_read


def create_conv2d_model(data_in):
    """
    Creates a model consisting of convolution layers, fully connected layers,
    and the output layer.
    Initial thoughts:
    - Have the same model, weights, biases process red, blue, and green values
    - Final output is the average of the outputs of the network for red, green,
        and blue channels
    - Might need separate learned weights/biases for each channel

    TODO:
        Figure out a better way to process all channels,
        conv2d layers, max_pooling2d layers, fully_connected layers at the end

    :param data_in: A Tensor of shape [?, 32, 32] representing the image at a
        single channel
    :return: A Tensor of shape [?, 10] representing the output of the network
        (10 possible classes)
    """
    data_in = tf.reshape(data_in, shape=[-1, cifar10_read.IMAGE_WIDTH,
                                         cifar10_read.IMAGE_HEIGHT, 1])
    return None
