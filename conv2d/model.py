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
    c1_2 = tf.layers.conv2d(inputs=c1_1, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_1_2")
    c1_3 = tf.layers.conv2d(inputs=c1_2, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_1_3")
    c1_4 = tf.layers.conv2d(inputs=c1_3, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_1_4")
    p1 = tf.layers.max_pooling2d(inputs=c1_4, pool_size=2, strides=2,
                                 padding="same", name="p2d_1")

    c2_1 = tf.layers.conv2d(inputs=p1, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_2_1")
    c2_2 = tf.layers.conv2d(inputs=c2_1, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_2_2")
    c2_3 = tf.layers.conv2d(inputs=c2_2, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_2_3")
    p2 = tf.layers.max_pooling2d(inputs=c2_3, pool_size=2, strides=2,
                                 padding="same", name="p2d_2")

    c3_1 = tf.layers.conv2d(inputs=p2, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_3_1")
    c3_2 = tf.layers.conv2d(inputs=c3_1, filters=N_FILTERS, kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_3_2")
    p3 = tf.layers.max_pooling2d(inputs=c3_2, pool_size=2, strides=2,
                                 padding="same", name="p2d_3")

    c4_1 = tf.layers.conv2d(inputs=p3, filters=(N_FILTERS / 2), kernel_size=4,
                            padding="same", activation=tf.nn.relu,
                            name="c2d_4_1")
    p4 = tf.layers.max_pooling2d(inputs=c4_1, pool_size=2, strides=2,
                                 padding="same", name="p2d_4")

    # p4.shape == (-1, 2, 2, N_FILTERS/2)
    # 2 == 32 / (2^4) where 32 = image width, height and 4 = pool layer count
    p4_flattened = tf.reshape(p4, shape=[-1, 2 * 2 * (N_FILTERS / 2)])
    d1 = tf.layers.dense(p4_flattened, units=2048, activation=tf.nn.relu,
                         use_bias=True)
    d2 = tf.layers.dense(d1, units=1024, activation=tf.nn.relu, use_bias=True)
    d3 = tf.layers.dense(d2, units=512, activation=tf.nn.relu, use_bias=True)
    d4 = tf.layers.dense(d3, units=256, activation=tf.nn.relu, use_bias=True)
    out = tf.layers.dense(d4, units=len(cifar10_read.LABEL_NAMES))
    return out
