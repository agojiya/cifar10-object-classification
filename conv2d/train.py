from os import path, makedirs
from random import shuffle

from fileio import cifar10_read, tfsaverutils
from conv2d import model

import tensorflow as tf
import numpy as np

SAVE_PATH = 'X:/cifar10/model/conv2d/model-epoch'
SAVE_DIR = '/'.join(SAVE_PATH.split('/')[0:-1])
if not path.isdir(SAVE_DIR):
    makedirs(SAVE_DIR)

N_EPOCHS = 10
BATCH_SIZE = 1024

in_image = tf.placeholder(dtype=tf.float32,
                          shape=[None, cifar10_read.IMAGE_WIDTH,
                                 cifar10_read.IMAGE_HEIGHT, 3],
                          name='in')
in_label = tf.placeholder(dtype=tf.float32,
                          shape=[None, len(cifar10_read.LABEL_NAMES)])

model_out = model.create_conv2d_model(data_in=in_image)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out,
                                                              labels=in_label))
optimizer = tf.train.AdamOptimizer().minimize(loss)

train_data = cifar10_read.get_data('train')
train_images, train_labels = train_data['images'], train_data['labels']
num_examples = len(train_images)

saver = tf.train.Saver(max_to_keep=None)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    saved_epochs = tfsaverutils.get_highest_epoch_saved(SAVE_DIR)
    N_TOTAL_EPOCHS = N_EPOCHS + saved_epochs
    if saved_epochs != 0:
        saver.restore(sess=session,
                      save_path=(SAVE_PATH + '-' + str(saved_epochs)))
        print('Loaded', saved_epochs, 'epochs of training')

    for epoch in range(N_EPOCHS):
        epoch_loss = 0

        zipped_data = list(zip(train_images, train_labels))
        shuffle(zipped_data)
        train_images, train_labels = zip(*zipped_data)

        for i in range(int(num_examples / BATCH_SIZE) + 1):
            n = min(BATCH_SIZE, num_examples - BATCH_SIZE * i)
            images = np.asarray(
                train_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            labels = np.asarray(
                train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            _, batch_loss = session.run([optimizer, loss],
                                        feed_dict={in_image: images,
                                                   in_label: labels})
            epoch_loss += batch_loss
        current_epoch = saved_epochs + epoch + 1
        if current_epoch % 5 == 0:
            saver.save(sess=session, save_path=SAVE_PATH,
                       global_step=current_epoch)
        print(current_epoch, '/', N_TOTAL_EPOCHS, 'epochs completed', '(',
              '{:.5e}'.format(epoch_loss / num_examples), ')')
