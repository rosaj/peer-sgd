import tensorflow as tf
from tensorflow import keras
import numpy as np

BATCH_SIZE = 128


def __load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    print("WARNING using  small nets")
    return (x_train, y_train), (x_test, y_test)


def all_data():
    (train_x, train_y), (test_x, test_y) = __load_dataset()

    ds = DataSource(train_x, train_y, test_x, test_y, 60000)
    return ds


def split_data(num_of_splits):
    (train_x, train_y), (test_x, test_y) = __load_dataset()

    train_x_split = np.array_split(train_x, num_of_splits)
    train_y_split = np.array_split(train_y, num_of_splits)

    test_x_split = np.array_split(test_x, num_of_splits)
    test_y_split = np.array_split(test_y, num_of_splits)

    datasets = []

    for i in range(num_of_splits):
        ds = DataSource(train_x_split[i], train_y_split[i], test_x_split[i], test_y_split[i])
        datasets.append(ds)

    return datasets


def scale(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def map_dataset(x, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(scale)
    ds = ds.batch(batch_size)
    return ds


class DataSource:

    def __init__(self, train_x, train_y, test_x, test_y, batch_size=BATCH_SIZE):

        self.train = map_dataset(train_x, train_y, batch_size)
        self.test = map_dataset(test_x, test_y, batch_size)

        self.batch_count = len(list(self.train))

        self.train_iter = iter(self.train)
        self.test_iter = iter(self.test)

    def next_train_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train)
            return next(self.train_iter)

    def next_test_batch(self):
        try:
            return next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test)
            return self.next_test_batch()
