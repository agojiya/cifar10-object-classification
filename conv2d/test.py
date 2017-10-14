from os import path, makedirs
from sys import exit

from fileio import cifar10_read, tfsaverutils
from conv2d import model

import tensorflow as tf
import numpy as np

SAVE_PATH = 'X:/cifar10/model/conv2d/model-epoch'
SAVE_DIR = '/'.join(SAVE_PATH.split('/')[0:-1])
if not path.isdir(SAVE_DIR):
    makedirs(SAVE_DIR)

BATCH_SIZE = 1024

in_image = tf.placeholder(dtype=tf.float32,
                          shape=[None, cifar10_read.IMAGE_WIDTH,
                                 cifar10_read.IMAGE_HEIGHT, 3],
                          name='in')

model_out = model.create_conv2d_model(data_in=in_image)

test_data = cifar10_read.get_data(train_set=False)
test_images, test_labels = test_data['images'], test_data['labels']
num_examples = len(test_images)

saver = tf.train.Saver()
saved_epochs = tfsaverutils.get_highest_epoch_saved(SAVE_DIR)
if saved_epochs == 0:
    exit('Could not find save file')
with tf.Session() as session:
    saver.restore(sess=session,
                  save_path=(SAVE_PATH + '-' + str(saved_epochs)))
    print('Loaded', saved_epochs, 'epochs of training')

    correct_classifications = 0
    for i in range(int(num_examples / BATCH_SIZE) + 1):
        n = min(BATCH_SIZE, num_examples - BATCH_SIZE * i)
        images = np.asarray(
            test_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        actual_labels = np.asarray(
            test_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        predicted_labels = session.run(model_out, feed_dict={in_image: images})

        for x in range(len(actual_labels)):
            if np.argmax(actual_labels[x]) == np.argmax(predicted_labels[x]):
                correct_classifications += 1

    print('Accuracy: {}%'.format(
        round(correct_classifications / num_examples * 100, ndigits=2)))
