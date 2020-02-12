from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple

# Helper libraries
import time
import datetime
from agent import Agent
import agent
import data_source
import plot
import model
from model import Model
from data_set import DataSet
from data_set import DataSource
from util import *
from scenario import *
import os
import glob


acc_name = "accuracy"
if hasattr(tf.compat.v1, "enable_eager_execution"):
    tf.compat.v1.enable_eager_execution()
    acc_name = "acc"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = int(time.time())
print(seed)

hs = {}

global data, initial_weights
initial_weights = Model.create_model().get_weights()


def set_model_v(v):
    model.MODEL_V = v
    global initial_weights
    initial_weights = Model.create_model().get_weights()


def plot_totals():
    if len(hs) > 1:
        plot.plot_accuracy_s(hs, "total_{}_{}".format(seed, hs.keys()))


def prepare_data():
    global data
    data = DataSet()
    data.shuffle_train()
    data.shuffle_test()


def augment_test():
    global data
    data = DataSet()
    data.test_data_augmentation(5000)
    data.shuffle()


def augment_102400():
    global data
    data = DataSet()
    data.data_augmentation(100 * 8 * 128 - 60000)
    data.test_data_augmentation(5000)
    data.shuffle()


def augment_102400x2():
    global data
    data = DataSet()
    data.data_augmentation(100 * 8 * 128 - 60000)
    data.test_data_augmentation(5000,
                                rotation_range=20,
                                zoom_range=0.1,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
    data.shuffle()


def augment_102400x5():
    global data
    data = DataSet()
    data.data_augmentation(100 * 8 * 128 - 60000)
    data.test_data_augmentation(5000,
                                rotation_range=50,
                                zoom_range=0.25,
                                width_shift_range=0.25,
                                height_shift_range=0.25)
    data.shuffle()


def train_batches_num():
    global data
    return round(data.train_size / 128)


def div_data(num_agents):
    return round(train_batches_num() / num_agents)


def new_data_source(batches_num):
    return DataSource(data, batches_num())


def new_agent(batches_num, alpha):
    global data
    return Agent(new_data_source(batches_num), alpha)


def model_acc(m):
    global data
    test_loss, test_acc = m.evaluate(data.test_x, data.test_y, verbose=0)
    return test_acc


def agent_acc(a):
    # global data
    # test_loss, test_acc = a.model.evaluate(data.test_x, data.test_y, verbose=0)
    # return test_acc
    return model_acc(a.model)


def default_network(epochs, steps_update=0.1):
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs, steps_update):
            self.epochs = epochs
            self.steps_update = round(steps_update * epochs * train_batches_num())
            self.prog_bar = tf.keras.utils.Progbar(100, stateful_metrics=["loss", "acc"])
            self.acc_h = {}
            self.iter = 0

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                self.prog_bar.update(epoch / self.epochs * 100,
                                     values=[("loss", logs["loss"]),
                                             ("acc", logs[acc_name])])

        def on_batch_end(self, batch, logs=None):
            self.iter += 1
            if self.iter % self.steps_update == 0:
                self.acc_h[self.iter] = model_acc(self.model)

    prepare_data()

    global data
    train_x, train_y = data.train_x, data.train_y  # data.next_train_batch(data.train_size)

    pc = ProgressCallback(epochs, steps_update)
    model = Model.create_model()
    model.fit(train_x,
              train_y,
              epochs=epochs,
              batch_size=128,
              verbose=0,
              callbacks=[pc])

    hs["default"] = pc.acc_h

    test_loss, test_acc = model.evaluate(data.test_x, data.test_y, verbose=0)
    print('\nTest accuracy:', test_acc)

    return model


def init_agents(agent_count, batches_num, default_weights=None):
    # data_sources = data_source.split_data(agent_count)
    agents = []

    for i in range(agent_count):
        a = new_agent(batches_num, 1 / agent_count)
        agents.append(a)

        if default_weights is not None:
            a.model.set_weights(default_weights)

    return agents


def agents_accuracy(agents):
    return np.mean(np.array([agent_acc(a) for a in agents]))


def models_mean_acc(models):
    return np.mean(np.array([model_acc(m) for m in models]))


def models_acc(models):
    weights = average_weights([m.get_weights() for m in models])

    model = Model.create_model()
    model.set_weights(weights)
    global data
    test_loss, test_acc = model.evaluate(data.test_x, data.test_y, verbose=0)
    return test_loss, test_acc


def avg_weights_acc(agents, test_x, test_y):
    # agents = np.array(agents)[np.random.choice(len(agents), size=10, replace=False)]

    agents_weights = [a.model.get_weights() for a in agents]
    weights = average_weights(agents_weights)

    model = Model.create_model()
    model.set_weights(weights)

    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    return test_loss, test_acc


knowledge_sharing_options = {
    'pm': lambda agent1, agent2: agent1.push_message(agent2),
    'kl': lambda agent1, agent2: agent1.kl(agent2)
}


def time_elapsed(start_time):
    elapsed = datetime.timedelta(seconds=(time.time() - start_time))
    elapsed = ':'.join(str(elapsed).split(':')[:2])
    return elapsed


def simulate(name,
             epochs,
             agent_count,
             p=0.02,
             scenario_func=interact,
             default_weights=False,
             steps_update=0,
             data_func=prepare_data,
             save_weights=False,
             model_v=model.MODEL_V,
             batches_num=None,
             legend_name=None):

    if batches_num is None:
        batches_num = lambda: div_data(agent_count)

    np.random.seed(seed)
    data_func()
    set_model_v(model_v)

    start_time = time.time()
    kt_func = knowledge_sharing_options[name]

    history = []
    agent_iter = {}
    # agent_active = {}
    active_agents = []
    # ag_test_acc = []

    global data
    test_x, test_y = data.test_x, data.test_y

    __def_weights = initial_weights if default_weights else None

    agents = init_agents(agent_count, batches_num, __def_weights)

    # batches_num = (int(60000 / data_source.BATCH_SIZE / agent_count) + 1)
    # batches_num = int(np.mean(np.array([len(a.dataset.train) for a in agents])))
    # print("batches_num: ", batches_num)
    # iter_num = batches_num * epochs
    # iter_num = round((batches_num * epochs) / agent_count)

    iter_num = round(((data.train_size / 128) * epochs) / agent_count)

    prog_bar = tf.keras.utils.Progbar(100, stateful_metrics=["hE", "loss", "acc"])

    print("Iterating", iter_num)

    if steps_update <= 0:
        steps_update = max(1, round(iter_num / 10))
    elif steps_update < 1:
        steps_update = max(1, round(iter_num * steps_update))

    print("Steps update: ", steps_update)

    SimInfo = namedtuple('SimInfo', ['agents', 'iter', 'def_weights'])

    total_train_iters = 0
    interactions = 0

    # for i in range(iter_num):
    # i = -1
    while round(total_train_iters / agent_count) < iter_num:
        i = round(total_train_iters / agent_count)
        # print(agents[0].model.model.optimizer.iterations.numpy(), agents[0].model.model.optimizer._decayed_lr('float32').numpy())

        # print(i)
        for a in agents:
            a.trained = False

        scenario_func(SimInfo(agents, i, __def_weights))

        ag = [a for a in agents if a.active and a.trained]
        ac = len(ag)

        total_train_iters += ac

        if ac > 0:
            for j, a in enumerate(ag):
                if draw(p):
                    r = choose(j, agent_count)
                    kt_func(a, agents[r])
                    interactions += 1

        # agent_active[round(total_train_iters / agent_count)] = ac

        if i % steps_update == 0 and ac > 0:
            avg_loss, avg_acc = avg_weights_acc(ag, test_x, test_y)  # if ac > 0 else (0, 0)

            history.append(avg_acc)
            agent_iter[round(total_train_iters / agent_count)] = avg_acc

            active_agents.append(ac)

            # test_acc = agents_accuracy(ag)
            # ag_test_acc.append(test_acc)

            prog_bar.update(i / iter_num * 100,
                            values=[("hE", round((time.time() - start_time) / 3600, 2)),
                                    ("loss", avg_loss),
                                    ("acc", avg_acc),
                                    # ("test_acc", test_acc)
                                    ])

    avg_loss, avg_acc = avg_weights_acc([a for a in agents if a.active], test_x, test_y)

    #  agent_iter[round(total_train_iters / agent_count)] = avg_acc

    plot.plot_accuracy({"accuracy": history, "online": active_agents},  # , "test_acc": ag_test_acc},
                       title=name + "_{}a_{}e_{}p_{:.3%}acc_{:.3%}tacc_{}_{}_{}"
                       .format(len(agents),
                               epochs,
                               p,
                               avg_acc,
                               agents_accuracy(agents),  # {:.3%}tacc_
                               time_elapsed(start_time),
                               scenario_func.__name__,
                               model.MODEL_V))

    plot_model_history_s(agents, name)

    if legend_name is None:
        legend_name = "{}_{}_{}a_{}e".format(name, default_weights, agent_count, epochs)

    hs[legend_name] = agent_iter
    # hs[legend_name+" active"] = agent_active

    print("TOTAL NUM OF ITERS: ", total_train_iters)
    print("INTERACTIONS: ", interactions)

    if save_weights:
        folder = "{}_{}_{}a_{}e_{}p_{}v_{}lr_{}iter_{}_{}".format(name, default_weights, agent_count, epochs, p,
                                                                  model.MODEL_V, model.LEARNING_RATE, total_train_iters,
                                                                  scenario_func.__name__, data_func.__name__)
        save_agents(agents, folder)


def plot_model_history_s(agents, proto):
    plot.plot_accuracy_s({i: acc for i, acc in
                          zip(list(range(len(agents))), [a.model.train_history.history[acc_name] for a in agents])},
                         proto + "_{:.3%}train__{:.3%}test_{:.3%}avg".format(
                             max([a.model.train_history.history[acc_name][-1] for a in agents if
                                  len(a.model.train_history.history[acc_name]) > 0]),
                             max([agent_acc(a) for a in agents]),
                             np.mean(np.array([agent_acc(a) for a in agents]))),
                         show_legend=False)


def save_agents(agents, folder):
    folder = 'drive/My Drive/models/m{}/'.format(model.MODEL_V) + folder.replace('.', '_')
    tf.io.gfile.makedirs(folder)

    models = [a.model.model for a in agents]
    for i, m in enumerate(models):
        m.save('{}/model_{}.h5'.format(folder, i))

    m = Model.create_model()
    m.set_weights(average_weights([m.get_weights() for m in models]))
    m.save('{}/model.h5'.format(folder))


def load_agents(folder):
    models = load_models(folder)
    agents = []

    for m in models:
        a = new_agent(1 / len(models))
        a.model.model = m
        agents.append(a)

    return agents


def load_model(folder):
    folder = 'drive/My Drive/models/m{}/'.format(model.MODEL_V) + folder
    m = tf.keras.models.load_model('{}/model.h5'.format(folder))
    return m


def load_models(folder):
    folder = 'drive/My Drive/models/m{}/'.format(model.MODEL_V) + folder

    num = len(glob.glob1(folder, "*.h5"))
    if tf.io.gfile.exists('{}/model.h5'.format(folder)):
        num -= 1

    models = []
    for i in range(num):
        m = tf.keras.models.load_model('{}/model_{}.h5'.format(folder, i))
        models.append(m)

    print(num, end=' ')
    return models


def convert_to_one():
    folder = 'drive/My Drive/models/m{}/'.format(model.MODEL_V)

    files = glob.glob1(folder, "*")

    for name in files:
        if name.__contains__("500a") or tf.io.gfile.exists('{}/{}/model.h5'.format(folder, name)):
            continue

        ms = load_models(name)
        aw = average_weights([m.get_weights() for m in ms])

        m = Model.create_model()
        m.set_weights(aw)
        m.save("{}/{}/model.h5".format(folder, name))
        print(len(ms))






