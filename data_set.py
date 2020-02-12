from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets.cifar10 as cifar10
import tensorflow.keras.utils as k_utils
import numpy as np

MNIST = (mnist, 28, 28, 1)
CIFAR10 = (cifar10, 32, 32, 3)
DATA_SET = MNIST

"""
global curr
curr = 0
"""


class DataSet:
    train_x, train_y, test_x, test_y = None, None, None, None
    train_size, test_size = 0, 0

    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = DATA_SET[0].load_data()
        """
        global curr
        ind = np.argwhere(self.train_y == curr)
        curr = (curr + 1)%10

        self.train_x = self.train_x[ind]
        self.train_y = self.train_y[ind]
        """
        # reshape
        self.train_x = self.train_x.reshape(self.train_x.shape[0], DATA_SET[1], DATA_SET[2], DATA_SET[3])
        self.test_x = self.test_x.reshape(self.test_x.shape[0], DATA_SET[1], DATA_SET[2], DATA_SET[3])
        # convert from int to float
        self.train_x = self.train_x.astype('float32')
        self.test_x = self.test_x.astype('float32')
        # rescale values
        self.train_x /= 255.0
        self.test_x /= 255.0
        # Save dataset sizes
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]
        # Create one hot array
        # self.train_y = k_utils.to_categorical(self.train_y, 10)
        # self.test_y = k_utils.to_categorical(self.test_y, 10)

    def data_augmentation(self, train_augment_size=40000,
                          rotation_range=10,
                          zoom_range=0.05,
                          width_shift_range=0.05,
                          height_shift_range=0.05,
                          horizontal_flip=False,
                          vertical_flip=False,
                          zca_whitening=True):

        image_generator = ImageDataGenerator(
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            data_format="channels_last",
            zca_whitening=zca_whitening)

        # fit data for zca whitening
        image_generator.fit(self.train_x, augment=True)

        # get transformed images
        randidx = np.random.randint(self.train_size, size=train_augment_size)
        x_augmented = self.train_x[randidx].copy()
        y_augmented = self.train_y[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(train_augment_size),
                                           batch_size=train_augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.train_x = np.concatenate((self.train_x, x_augmented))
        self.train_y = np.concatenate((self.train_y, y_augmented))

        self.train_size = self.train_x.shape[0]

    def test_data_augmentation(self, test_augment_size=6000,
                               rotation_range=10,
                               zoom_range=0.05,
                               width_shift_range=0.05,
                               height_shift_range=0.05,
                               horizontal_flip=False,
                               vertical_flip=False,
                               zca_whitening=True):

        image_generator = ImageDataGenerator(
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            data_format="channels_last",
            zca_whitening=zca_whitening)

        # fit data for zca whitening
        image_generator.fit(self.test_x, augment=True)

        # get transformed images
        randidx = np.random.randint(self.test_size, size=test_augment_size)
        x_augmented = self.test_x[randidx].copy()
        y_augmented = self.test_y[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(test_augment_size),
                                           batch_size=test_augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.test_x = np.concatenate((self.test_x, x_augmented))
        self.test_y = np.concatenate((self.test_y, y_augmented))

        self.test_size = self.test_x.shape[0]

    def next_train_batch(self, batch_size):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.train_x[randidx]
        epoch_y = self.train_y[randidx]
        return epoch_x, epoch_y

    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.test_x[randidx]
        epoch_y = self.test_y[randidx]
        return epoch_x, epoch_y

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.train_x = self.train_x[indices]
        self.train_y = self.train_y[indices]

    def shuffle_test(self):
        indices = np.random.permutation(self.test_size)
        self.test_x = self.test_x[indices]
        self.test_y = self.test_y[indices]

    def shuffle(self):
        self.shuffle_test()
        self.shuffle_train()


class DataSource:

    def __init__(self, mnist_generator, num_of_train_batches, num_of_test_batches=1, batch_size=128):
        # mnist_generator = DataSet()

        self.train = [mnist_generator.next_train_batch(batch_size) for _ in range(num_of_train_batches)]
        self.test = [mnist_generator.next_test_batch(batch_size) for _ in range(num_of_test_batches)]

        self.batch_count = len(self.train)

        self.train_iter = iter(self.train)
        self.test_iter = iter(self.test)

        self.__cur = []

    def current_train_batch(self):
        return self.__cur

    def next_train_batch(self):
        try:
            self.__cur = next(self.train_iter)
            return self.__cur
        except StopIteration:
            self.train_iter = iter(self.train)
            return self.next_train_batch()

    def next_test_batch(self):
        try:
            return next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test)
            return self.next_test_batch()
