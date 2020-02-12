import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

from model_history import ModelHistory

MODEL_V = 1
LEARNING_RATE = 0.01
DECAY = 0.0  # 1e-4


class Model:
    def __init__(self, batches_in_epoch):
        self.model = Model.create_model()

        self.train_history = ModelHistory(self.model.metrics_names)
        self.batch_train_count = 0
        self.batches_in_epoch = batches_in_epoch

    def train_on_batch(self, train_x, train_y):
        self.model.reset_metrics()

        train_metrics = self.model.train_on_batch(train_x, train_y)

        self.batch_train_count += 1

        if self.is_epoch_end():
            self.train_history.update(train_metrics)

        return train_metrics

    def is_epoch_end(self):
        return self.batch_train_count % self.batches_in_epoch == 0

    def evaluate(self, x, y, verbose=0):
        return self.model.evaluate(x, y, verbose=verbose)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def clone_model(self):
        # clone_model = keras.models.clone_model(model)
        clone_model = Model.create_model()
        # ModelSet.__compile_model(clone_model)
        #  clone_model.build()
        clone_model.set_weights(self.model.get_weights())
        return clone_model

    @staticmethod
    def __small_model():
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model

    @staticmethod
    def __big_model():
        model = keras.Sequential([
            layers.Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dropout(0.2),
            layers.Dense(10, activation=tf.nn.softmax)
        ])
        return model

    @staticmethod
    def le_net():
        model = keras.Sequential([
            layers.Conv2D(20, 5, padding="same", input_shape=(28, 28, 1)),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(50, 5, padding="same"),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(500),
            layers.Activation("relu"),
            layers.Dense(10),
            layers.Activation("softmax"),
        ])
        return model

    @staticmethod
    def __cifar10_model():
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')])
        return model

    @staticmethod
    def __big_cifar10_model():
        model = keras.Sequential([
            layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            layers.Dropout(0.25),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, input_shape=(256,)),
            layers.Dense(10),
            layers.LeakyReLU(0.1),
            layers.Activation('softmax')

        ])
        return model

    @staticmethod
    def create_model():
        m_v = {1: Model.__small_model,
               2: Model.__big_model,
               3: Model.le_net,
               4: Model.__cifar10_model,
               5: Model.__big_cifar10_model}

        model = m_v[MODEL_V]()

        Model.compile_model(model)
        return model

    @staticmethod
    def compile_model(model):
        """ # default params for Adam optimizer
        __init__(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam',
            **kwargs
        )
        """
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=DECAY)

        compute_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # compute_loss = keras.losses.KLDivergence()
        # compute_accuracy = keras.metrics.SparseCategoricalAccuracy()

        model.compile(optimizer=optimizer,
                      loss=compute_loss,
                      metrics=['accuracy'])
