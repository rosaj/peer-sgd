from model import Model
from util import *
import numpy as np

ACCEPT_PCT = 0.5

global MAX
MAX = ACCEPT_PCT


class Agent:

    def __init__(self, data_source, alpha):
        self.dataset = data_source
        self.model = Model(self.dataset.batch_count)

        self.alpha = alpha
        # self.accept_diff = max(np.random.rand().real, 0.15)
        # self.accept_diff = 1 - (data_source.batch_count / 120)
        self.accept_diff = ACCEPT_PCT
        self.active = True
        self.trained = False

    def train(self):
        if self.active:
            train_x, train_y = self.dataset.next_train_batch()
            self.model.train_on_batch(train_x, train_y)
            self.trained = True

    def __update_weights(self, mj):
        (x, y) = self.dataset.next_train_batch()

        mj = mj.model
        mi = self.model.model

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits_i = mi(x)
            logits_j = mj(x)

            loss_i = mi.loss(y, logits_i)
            loss_j = mj.loss(y, logits_j)

            loss_diff = abs((loss_i.numpy() - loss_j.numpy()) / loss_i.numpy())

            if loss_diff > self.accept_diff:  # and loss_j > loss_i:
                print(" ", loss_diff)
                return

            kl_loss = kl_loss_compute(logits_i, logits_j)

            i = 1 - loss_i / (loss_i + loss_j)
            j = 1 - loss_j / (loss_i + loss_j)

            mi.set_weights(add_weights((multiply_weights_with_num(mi.get_weights(), i.numpy()),
                                        multiply_weights_with_num(mj.get_weights(), j.numpy()))))

            loss = loss_i + kl_loss

        grads = tape.gradient(loss, mi.trainable_variables)
        # print(grads)

        mi.optimizer.apply_gradients(zip(grads, mi.trainable_variables))

    def __go_sgd(self, xj, aj):
        xi = self.model
        ai = self.alpha
        w1 = multiply_weights_with_num(xj.get_weights(), (aj / (aj + ai)))
        w2 = multiply_weights_with_num(xi.get_weights(), (ai / (ai + aj)))
        w = add_weights((w1, w2))
        xi.set_weights(w)

    def push_message(self, agent2):
        self.alpha /= 2.0
        if agent2.active:
            agent2.__go_sgd(self.model, self.alpha)
            agent2.alpha += self.alpha

    def kl(self, agent2):
        if agent2.active:
            agent2.__update_weights(self.model)
