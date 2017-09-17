import numpy as np

BASE_DIR = 'X:/cifar10'
FILES = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']


def unpickle(file):
    """
    Provided by https://www.cs.toronto.edu/~kriz/cifar.html
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


batches_meta_unpickled = unpickle(BASE_DIR + '/' + FILES[0])
NUM_CASES_PER_BATCH = batches_meta_unpickled[b'num_cases_per_batch']
LABEL_NAMES = batches_meta_unpickled[b'label_names']
IMAGE_WIDTH = IMAGE_HEIGHT = 32


def get_data(train_set=True):
    """
    Provides a dictionary containing both the images (RGB) and their respective labels. Elements in 'images' are
    shaped: (<num_images>, 3, 1024) where 3 represents the color channels and 1024 represents intensity values for each
    pixel in the respective channel (32 * 32 = 1024).

    :param train_set: Boolean indicating whether the training data should be loaded (True) or the test data (False).
    :return: A dictionary ('labels', 'images') where the nth label corresponds to the nth image.
    """
    unpickled_target_files = [unpickle(BASE_DIR + '/' + file) for file in (FILES[1:6] if train_set else FILES[6:])]

    output = {'labels': [],
              'images': np.empty(shape=(NUM_CASES_PER_BATCH, 3, IMAGE_WIDTH * IMAGE_HEIGHT), dtype=np.uint8)}
    for unpickled_file in unpickled_target_files:
        output['labels'] += unpickled_file[b'labels']
        output['images'] += np.reshape(unpickled_file[b'data'], (NUM_CASES_PER_BATCH, 3, IMAGE_WIDTH * IMAGE_HEIGHT))
    return output

# Todo: Preview images to make sure they're loading correctly
