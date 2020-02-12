"""
from util import *
import numpy as np

i = 469
ac = 100
# 93615 1049 878 936.15
# 9220 956 850 922.0

# 94014 1007 862 940.14
# 9406 971 916 940.6
# i = 4688
# ac = 10

e = 100

c = 0
iter_num = round((60000/128) * e)

sl = []
ap = []
rl = []

for _ in range(ac):
    sl.append(0)
    rl.append(0)
    ap.append(np.random.randint(2, 16))

amax = max(ap)

for _ in range(iter_num):
    for j in range(ac):
        # p = 0.02
        p = ap[j] / amax
        if draw( p ):
            c+=1
            sl[j] += 1
            s = choose(j, ac)
            rl[s] += 1

print(c, max(sl), min(sl), np.mean(np.array(sl)))
print(max(rl), min(rl), np.mean(np.array(rl)))
# 0.02
# 1004 879 941.02
# 1011 872 941.02

# other
# 46875 6179 25646.59
# 26249 25180 25646.59

#send 2809183 46875 6084 28091.83
"""

"""
            # loss = loss_i + kl_loss_compute(logits_i, logits_j)
            # loss_i + kl_loss * j
            # loss_i * j + loss_j * i
            # loss_i * j + loss_j * i * kl_loss / 2.0
            # loss = loss_i * i + loss_j * j * kl_loss / 2.0

            # loss = loss_i * i + loss_j * j + kl_loss
            # loss = loss_i * i + loss_j * j + kl_loss * j

            # loss = (loss_i * i + loss_j * j) / 2.0 + kl_loss

            # loss = loss_i
            # loss = loss_i * i
            # loss = (loss_i + loss_j) / 2.0
            # loss = (loss_i + loss_j) / 2.0 + kl_loss  # -> this
            # loss = (loss_i + loss_j) / 2.0 + kl_loss / 2.0

            # loss = loss_i * i + loss_j * j + kl_loss
            # loss = loss_i * j + loss_j * i + kl_loss

            # loss = (loss_i + loss_j) / 2.0 + kl_loss * i
            # loss = (loss_i + loss_j) / 2.0

            #
            # loss = (loss_i * i + loss_j * j) / 2.0
            # loss = (loss_i * i + loss_j * j) / 2.0 + kl_loss
            # loss = (loss_i * i + loss_j * j) / 2.0 + kl_loss / 2.0

            # loss = (loss_i * i + loss_j * j) / 2.0 # + kl_loss / 2.0
"""

"""
    def __test(self, mj):
        (x, y) = self.dataset.next_train_batch()

        mj = mj.model
        mi = self.model.model

        y = tf.convert_to_tensor(y, dtype=tf.float32)
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits_i = mi(x)
            logits_j = mj(x)

            # loss_i = mi.loss(y_true=y, y_pred=tf.convert_to_tensor(tf.argmax(logits_i, axis=1).numpy(), dtype=tf.float32))
            loss_i = mi.loss(y, logits_i)
            loss_j = mj.loss(y, logits_j)

            kl = tf.keras.losses.KLDivergence()
            k_loss_i = kl(y_true=y, y_pred=tf.convert_to_tensor(tf.argmax(logits_i, axis=1).numpy(), dtype=tf.float32))
            k_loss_j = kl(y_true=y, y_pred=tf.convert_to_tensor(tf.argmax(logits_j, axis=1).numpy(), dtype=tf.float32))
            loss_dif = kl(y_true=tf.convert_to_tensor(tf.argmax(logits_i, axis=1).numpy(), dtype=tf.float32),
                          y_pred=tf.convert_to_tensor(tf.argmax(logits_j, axis=1).numpy(), dtype=tf.float32))

            k_loss_j += loss_dif / 2.0
            k_loss_i += loss_dif / 2.0

            xi = 1 - (k_loss_i / (k_loss_i + k_loss_j))  # .numpy()
            xj = 1 - (k_loss_j / (k_loss_i + k_loss_j))  # .numpy()

            # loss = loss_i * xi + loss_j * xj

            loss = loss_i + xi + loss_j * xj
            # if k_loss_i == 0.0:
            #    loss = loss_i

            if np.isnan(xi) or np.isnan(xj) or np.isnan(loss):
                return

            print(" ", loss.numpy(), loss_i.numpy(), k_loss_i.numpy(), loss_dif.numpy())

            # loss_i += loss_dif / 2.0
            # loss_j += loss_dif / 2.0
            # loss_i += loss_dif

            # print(loss_i.numpy(), loss_j.numpy(), xi.numpy(), xj.numpy())

            mi.set_weights(add_weights((multiply_weights_with_num(mi.get_weights(), xi.numpy()),
                                        multiply_weights_with_num(mj.get_weights(), xj.numpy()))))

            # loss = loss_i * xi + loss_j * xj
            # loss = loss_i + loss_dif / 2.0

            # loss = loss_i * xi + loss_j * xj   # + (loss_i + loss_j) / (loss_i * loss_j)
            # print(loss)
            # loss = loss_i
            # loss = tf.identity(tf.convert_to_tensor(loss_dif))
            # loss_i = tf.convert_to_tensor(2.34)

        grads = tape.gradient(loss, mi.trainable_variables)
        # print(grads)

        mi.optimizer.apply_gradients(zip(grads, mi.trainable_variables))


"""


"""


    def __kl(self, mj):

        (x, y) = self.dataset.next_train_batch()

        mj = mj.model
        mi = self.model.model

        #
        data = MNIST()
        ds = MnistDataSource(data, 8, 1, 128)
        x, y = ds.next_train_batch()
        mi = Model.create_model()
        mj = Model.create_model()
        mi.train_on_batch(x, y)
        mj.train_on_batch(x, y)
        #

        y_pred_i = mi.predict(x)
        y_pred_j = mj.predict(x)

        kl = tf.keras.losses.KLDivergence()
        loss_i = kl(y_true=tf.convert_to_tensor(y, dtype=tf.float32),
                    y_pred=tf.convert_to_tensor(tf.argmax(y_pred_i, axis=1).numpy(), dtype=tf.float32))
        loss_j = kl(y_true=tf.convert_to_tensor(y, dtype=tf.float32),
                    y_pred=tf.convert_to_tensor(tf.argmax(y_pred_j, axis=1).numpy(), dtype=tf.float32))
        loss_dif = kl(y_true=tf.convert_to_tensor(tf.argmax(y_pred_i, axis=1).numpy(), dtype=tf.float32),
                      y_pred=tf.convert_to_tensor(tf.argmax(y_pred_j, axis=1).numpy(), dtype=tf.float32))

        # loss_dif /= 2.0
        loss_i += loss_dif
        loss_j += loss_dif

        if loss_i == 0.0 or loss_j == 0.0 or loss_i == loss_j:
            return

        xi = 1 - (loss_i / (loss_i + loss_j))  # .numpy()
        xj = 1 - (loss_j / (loss_i + loss_j))  # .numpy()

        print(loss_i.numpy(), loss_j.numpy(), loss_dif.numpy(), xi.numpy(), xj.numpy())

        mi.set_weights(add_weights((multiply_weights_with_num(mi.get_weights(), xi.numpy()),
                                    multiply_weights_with_num(mj.get_weights(), xj.numpy()))))
"""




"""
def avg_history(agents, epochs, p, acc):
    f = []
    l = []
    for a in agents:
        ar = np.array(a.model_set.models[0].train_history.history["accuracy"])
        print(len(ar))
        f.append(ar)
        l1 = np.array(a.model_set.models[0].train_history.history["loss"])
        l.append(l1)

    f = [np.mean(k) for k in zip(*f)]
    l = [np.mean(k) for k in zip(*l)]
    # plot.plot_accuracy({"accuracy": f, "loss": l}, title="{}a_{}e_{}p_{:.3%}acc".format(len(agents), epochs, p, acc))
    print(len(f))
    plot.plot_accuracy({"accuracy": f, "loss": l}, title="{}a_{}e_{}p_{}acc".format(len(agents), epochs, p, int(acc)))
"""

"""
dataset = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images.shape

len(train_labels)

train_labels

len(test_labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])
test_labels[0]
"""


"""
train_data = train_dataset()
BUFFER_SIZE = 10  # Use a much larger value for real code.
BATCH_SIZE = 64
NUM_EPOCHS = 5

STEPS_PER_EPOCH = 1

train_data = train_data.take(STEPS_PER_EPOCH)
# test_data = test_data.take(STEPS_PER_EPOCH)

# image_batch, label_batch = next(iter(train_data))


metrics_names = model.metrics_names
model.reset_metrics()

for image_batch, label_batch in train_data:
    print(len(image_batch))

for image_batch, label_batch in train_data:
    result = model.train_on_batch(image_batch, label_batch)
    print("train: ",
          "{}: {:.3f}".format(metrics_names[0], result[0]),
          "{}: {:.3f}".format(metrics_names[1], result[1]))

    with tf.GradientTape() as tape:
        logits = model(image_batch)
        loss = compute_loss(label_batch, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(label_batch, logits)

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset():
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


"""

"""




class Models:

    def __init__(self):
        pass


model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
model.build()

# opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

optimizer = tf.keras.optimizers.Adam()

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss


@tf.function
def train(model, optimizer):
    (train_c, train_l), _= tf.keras.datasets.mnist.load_data()

    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in zip(train_c, train_l):
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if step % 10 == 0:
            tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
    return step, loss, accuracy


step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())

"""

"""
    def train(self):
        for i in range(0, self.model_count):
            m = self.models[i]
            m.fit(self.data_source.train_images, self.data_source.train_labels, epochs=1)

    def evaluate(self):
        for i in range(0, self.model_count):
            m = self.models[i]
            test_loss, test_acc = m.evaluate(self.data_source.test_images, self.data_source.test_labels, verbose=2)

            print('\nTest accuracy:', test_acc)

    def __create_default_models(self):
        self.models = []

        for i in range(0, self.model_count):
            self.models.append(self.__create_model())

    @staticmethod
    def __create_model():
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        model.compile(optimizer='adam',  # 'adam'
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
"""





"""


def calc_grads(a):
    model = a.model_set.models[0]
    x, y = a.model_set.dataset.next_train_batch()
    compute_loss = model.loss

    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)

    model.metrics[0](y, logits)
    # print(model.metrics[0].result())

    return grads


def avg_grads(grads):
    g = grads[0]
    for i in range(1, len(grads)):
        grad = grads[i]

        for j in range(len(g)):
            g[j] += grad[j]

    for i in range(len(g)):
        g[i] /= len(grads)

    return g


def apply_grads(a, grads):
    model = a.model_set.models[0]
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))


def test_grads(epochs):
    agent_count = 20
    model_count = 1
    agents = []

    data_sources = data_source.split_data(agent_count)

    for i in range(agent_count):
        a = agent.Agent(data_sources[i], model_count)
        agents.append(a)

    weights = a.model_set.models[0].get_weights()
    for d in range(agent_count):
        agents[d].model_set.models[0].set_weights(weights)

    random.seed(time.process_time())

    # g1 = calc_grads(a)
    # g2 = calc_grads(a1)
    # gr = avg_grads([g1, g2])
    # apply_grads(a, gr)
    # apply_grads(a1, gr)

    for i in range(int(60000 / data_source.BATCH_SIZE / agent_count) * epochs + 1):
        grads = []
        for a in agents:
            grads.append(calc_grads(a))

        gr = avg_grads(grads)

        for a in agents:
            apply_grads(a, gr)

    ds = data_source.all_data()
    test_x, test_y = ds.next_test_batch()

    sum = 0
    for a in agents:
        for model in a.model_set.models:
            #  acc = a.model_set.accuracy_on_test(model).numpy()
            test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)

            sum += test_acc
            print("Accuracy on the end: {:.3%}".format(test_acc))

    print("Avg: ", sum / (agent_count * model_count) * 100)




"""
