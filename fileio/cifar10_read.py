import numpy as np

BASE_DIR = 'X:/cifar10'
FILES = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3',
         'data_batch_4', 'data_batch_5', 'test_batch']


def unpickle(file):
    """
    Provided by https://www.cs.toronto.edu/~kriz/cifar.html
    """
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

batches_meta_unpickled = unpickle(BASE_DIR + '/' + FILES[0])
NUM_CASES_PER_BATCH = batches_meta_unpickled[b'num_cases_per_batch']
LABEL_NAMES = [s.decode('UTF-8') for s in
               batches_meta_unpickled[b'label_names']]
IMAGE_WIDTH = IMAGE_HEIGHT = 32


def get_data(train_set=True):
    """
    Provides a dictionary containing both the images (RGB) and their respective
    labels. Elements in 'images' are shaped: (<num_images>, IMAGE_WIDTH,
    IMAGE_HEIGHT, 3) where 3 represents the color channels and 1024 represents
    intensity values for each pixel in the respective channel (32 * 32 = 1024).

    :param train_set: Boolean indicating whether the training data should be
        loaded (True) or the test data (False).
    :return: A dictionary ('labels', 'images') where the nth label corresponds
        to the nth image.
    """
    unpickled_target_files = [unpickle(BASE_DIR + '/' + file) for file in
                              (FILES[1:6] if train_set else FILES[6:])]

    output = {'labels': np.empty(shape=(NUM_CASES_PER_BATCH,
                                        len(LABEL_NAMES))),
              'images': np.empty(
                  shape=(NUM_CASES_PER_BATCH, IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                  dtype=np.uint8)}
    for unpickled_file in unpickled_target_files:
        labels = unpickled_file[b'labels']
        one_hot_label = np.zeros(shape=(NUM_CASES_PER_BATCH, len(LABEL_NAMES)))
        one_hot_label[np.arange(NUM_CASES_PER_BATCH), labels] = 1
        output['labels'] = np.append(output['labels'], one_hot_label, axis=0)

        current_images = unpickled_file[b'data']
        # Reshape using fortran-like indexing (order='F') to ensure that the
        # values at each index are correct (default order='C' is not how the
        # data was stored in the dataset)
        current_images_reshaped = current_images.reshape((NUM_CASES_PER_BATCH,
                                                          IMAGE_WIDTH,
                                                          IMAGE_HEIGHT, 3),
                                                         order='F')
        output['images'] = np.append(output['images'],
                                     current_images_reshaped, axis=0)
    # Remove the first NUM_CASES_PER_BATCH (10000) empty entries
    output['labels'] = np.delete(output['labels'],
                                 np.s_[:NUM_CASES_PER_BATCH], 0)
    output['images'] = np.delete(output['images'],
                                 np.s_[:NUM_CASES_PER_BATCH], 0)

    # Images are sideways (maybe a byproduct of fortran-like order in the
    # reshape?)
    output['images'] = np.rot90(output['images'], k=-1, axes=(1, 2))
    return output


if __name__ == "__main__":
    data = get_data(train_set=True)

    import cv2
    from random import randrange

    while True:
        index = randrange(0, stop=len(data['labels']))
        print(LABEL_NAMES[np.argmax(data['labels'][index])])

        bigger_image = np.repeat(np.repeat(data['images'][index], 4, axis=0),
                                 4, axis=1)
        cv2.imshow('test', cv2.cvtColor(bigger_image, cv2.COLOR_RGB2BGR))

        # Any key to show another random image, space-bar to exit
        if cv2.waitKey() & 0xFF == 32:
            break
