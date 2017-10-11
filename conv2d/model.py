import tensorflow as tf
from fileio import cifar10_read

N_FILTERS = 64


def create_conv2d_model(data_in):
    """
    Creates a model consisting of convolution layers, fully connected layers,
    and the output layer. Accepts the entire image with all three channels.

    TODO:
        conv2d layers,
        fully_connected layers at the end

    :param data_in: A Tensor of shape [?, 32, 32, 3] representing the image at
        a single channel
    :return: A Tensor of shape [?, 10] representing the output of the network
        (10 possible classes)
    """
    c1_1 = tf.layers.conv2d(inputs=data_in, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_1_1")
    # .
    # .
    # .
    p1 = tf.layers.max_pooling2d(inputs=c1_1, pool_size=2, strides=2,
                                 padding="same", name="p2d_1")

    c2_1 = tf.layers.conv2d(inputs=p1, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_2_1")
    # .
    # .
    # .
    p2 = tf.layers.max_pooling2d(inputs=c2_1, pool_size=2, strides=2,
                                 padding="same", name="p2d_2")

    c3_1 = tf.layers.conv2d(inputs=p2, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_3_1")
    # .
    # .
    # .
    p3 = tf.layers.max_pooling2d(inputs=c3_1, pool_size=2, strides=2,
                                 padding="same", name="p2d_3")

    c4_1 = tf.layers.conv2d(inputs=p3, filters=(N_FILTERS / 2), kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_4_1")
    # .
    # .
    # .
    p4 = tf.layers.max_pooling2d(inputs=c4_1, pool_size=2, strides=2,
                                 padding="same", name="p2d_4")

    # p4.shape == (-1, 2, 2, N_FILTERS/2)
    # 2 == 32 / (2^4) where 32 = image width, height and 4 = pool layer count
    # Todo: Fully connected layers
    return None
