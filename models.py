from model import Model
from util import *


class ModelSet:

    def __init__(self, model_count, dataset):
        self.dataset = dataset
        self.__create_default_models(model_count)

    def __create_default_models(self, model_count):
        self.models = [Model(self.dataset.batch_count) for _ in range(model_count)]

    def train_batch(self):
        image_batch, label_batch = self.dataset.next_train_batch()

        for model in self.models:
            model.train_on_batch(image_batch, label_batch)

    def swap(self, models):
        (x, y) = self.dataset.next_test_batch()
        mj = models[0]
        mi = self.models[0]
        loss_j, acc_j = mj.evaluate(x, y)
        loss_i, acc_i = mi.evaluate(x, y)

        if acc_j > acc_i:
            mi.set_weights(mj.get_weights())

    def transfer(self, models):
        (x, y) = self.dataset.next_test_batch()

        mj = models[0]
        mi = self.models[0]
        loss_j, acc_j = mj.evaluate(x, y)
        loss_i, acc_i = mi.evaluate(x, y)

        acc_change = abs(acc_i - acc_j)

        # if acc_j > acc_i:
        if acc_change < 0.01 or acc_j > acc_i:
            # j = acc_i/acc_j
            # i = 1 - j

            j = acc_j / (acc_i + acc_j)
            i = acc_i / (acc_i + acc_j)

            w1 = multiply_weights_with_num(mj.get_weights(), j)
            w2 = multiply_weights_with_num(mi.get_weights(), i)
            w = add_weights((w1, w2))
            mi.set_weights(w)

    def go_sgd(self, models, aj, ai):
        xi = self.models[0]
        xj = models[0]

        # w = (aj / (aj + ai)) * xj.get_weights() + (ai / (ai + aj)) * xi.get_weights()
        w1 = multiply_weights_with_num(xj.get_weights(), (aj / (aj + ai)))
        w2 = multiply_weights_with_num(xi.get_weights(), (ai / (ai + aj)))
        w = add_weights((w1, w2))
        xi.set_weights(w)


"""


    @staticmethod
    def multiply_weights(weights, num):
        for i in range(len(weights)):
            weights[i] *= num
        return weights

    @staticmethod
    def add_weights(weight1, weight2):
        w = []
        for i, w1 in enumerate(weight1):
            w.append(w1 + weight2[i])

        return w

    def average_models(self, models):
        for i, m in enumerate(models):
            w1 = self.models[i].model.get_weights()
            w2 = m.model.get_weights()
            w = ModelSet.average_weights(w1, w2)
            self.models[i].model.set_weights(w)

    @staticmethod
    def average_weights(weights1, weights2):
        weights = []

        for y, w1 in enumerate(weights1):
            w2 = weights2[y]

            w = (w1 + w2) / 2.0

            weights.append(w)

        return weights

    def accuracy_on_test_batch(self, model):
        test_accuracy = tf.keras.metrics.Accuracy()

        (x, y) = self.dataset.next_test_batch()
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

        return test_accuracy.result()

    def accuracy_on_test(self, model):
        test_accuracy = tf.keras.metrics.Accuracy()

        for (x, y) in self.dataset.test:
            logits = model(x)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            test_accuracy(prediction, y)

        return test_accuracy.result()
        
"""

"""
if __name__ == '__main__':
    net_count = 2
    ms = ModelSet(net_count, data_source.DataSource())
    ms.train_batch()

"""

"""
    @staticmethod
    def kl_loss_compute(logits1, logits2):
        # KL loss
        
        pred1 = tf.nn.softmax(logits1)
        pred2 = tf.nn.softmax(logits2)

        loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

        return loss
"""

"""

    def accuracy_on_test_batch(self, model):
        test_accuracy = tf.keras.metrics.Accuracy()

        (x, y) = self.dataset.next_test_batch()
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

        return test_accuracy.result()

    def accuracy_on_test(self, model):
        test_accuracy = tf.keras.metrics.Accuracy()

        for (x, y) in self.dataset.test:
            logits = model(x)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            test_accuracy(prediction, y)

        return test_accuracy.result()

    def accept_new_models(self, new_models):
        self.exchange_all_weights(new_models)

    def exchange_all_weights(self, new_models):

        weights = new_models[0].get_weights()
        for i, m in enumerate(new_models):
            if i == 0:
                continue
            for y, w in enumerate(m.get_weights()):
                weights[y] += w

        for i in range(len(weights)):
            weights[i] /= len(new_models)

        for my in self.models:
            wm = []
            for y, w in enumerate(my.get_weights()):

                wf = (w + weights[y]) / 2.0
                wm.append(wf)

        my.set_weights(wm)

    def exchange_weights(self, new_models):
        d = {}

        for i, model in enumerate(new_models):
            acc = self.accuracy_on_test_batch(model)
            acc = acc.numpy()
            d[acc] = i

        mine_d = {}
        for i, model in enumerate(self.models):
            acc = self.accuracy_on_test_batch(model)
            acc = acc.numpy()
            mine_d[acc] = i

        transfer_pct = 50

        transfer_count = int(transfer_pct / 100 * len(self.models))

        for i in range(transfer_count):
            if len(d) == 0 or len(mine_d) == 0:
                break
            max_d = max(d)
            min_mine_d = min(mine_d)

            if max_d <= min_mine_d:
                break

            nm = d[max_d]
            new_m = new_models[nm]
            my_m = self.models[mine_d[min_mine_d]]

            weights = []

            for y, w1 in enumerate(my_m.get_weights()):
                w2 = new_m.get_weights()[y]

                w = (w1 + w2) / 2.0

                weights.append(w)

            my_m.set_weights(weights)

            del d[max_d]
            del mine_d[min_mine_d]

    def both_average_weights_update(self, new_models):

        for i in range(len(new_models)):
            m1 = self.models[i]
            m2 = new_models[i]

            weights = []

            for y, w1 in enumerate(m1.get_weights()):
                w2 = m2.get_weights()[y]

                w = (w1 + w2) / 2.0

                weights.append(w)

            m1.set_weights(weights)
            m2.set_weights(weights)

    def average_weights_update(self, new_models):

        for i in range(len(new_models)):
            m1 = self.models[i]
            m2 = new_models[i]

            weights = []

            for y, w1 in enumerate(m1.get_weights()):
                w2 = m2.get_weights()[y]

                w = (w1 + w2) / 2.0

                weights.append(w)

            m1.set_weights(weights)

    def clone_best_models(self, new_models):
        d = {}

        for i, model in enumerate(new_models):
            acc = self.accuracy_on_test_batch(model)
            acc = acc.numpy()
            d[acc] = i
            # print("Accuracy of new model on my test data: {:.3%}".format(acc))

        mine_d = {}
        for i, model in enumerate(self.models):
            acc = self.accuracy_on_test_batch(model)
            acc = acc.numpy()
            mine_d[acc] = i
            # print("Accuracy of my model on my test data: {:.3%}".format(acc))

        transfer_pct = 50

        transfer_count = int(transfer_pct / 100 * len(self.models))

        for i in range(transfer_count):
            if len(d) == 0 or len(mine_d) == 0:
                break
            max_d = max(d)
            min_mine_d = min(mine_d)

            if max_d <= min_mine_d:
                break

            nm = d[max_d]
            new_m = ModelSet.clone_model(new_models[nm])
            self.models[mine_d[min_mine_d]] = new_m

            # print("Cloned: {:.3%}  Original: {:.3%}".format(self.accuracy_on_test(new_m), self.accuracy_on_test(new_models[nm])))

            del d[max_d]
            del mine_d[min_mine_d]


"""
