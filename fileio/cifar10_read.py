
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

print(unpickle(BASE_DIR + '/' + FILES[0]))

# Todo: Get and return a simplified dictionary with labels and image data only
