import numpy as np
import tensorflow as tf


def calc_lr(lr, decay, iters):
    return round(lr * 1/(1 + decay * iters), ndigits=9)


def multiply_weights_with_num(weights, num):
    for i in range(len(weights)):
        weights[i] *= num
    return weights


def add_weights(model_weights):
    return __apply_on_weights(model_weights, np.sum)


def average_models(models1, models2):
    for i, m in enumerate(models2):
        w1 = models1[i].model.get_weights()
        w2 = m.model.get_weights()
        w = average_weights((w1, w2))
        models1[i].model.set_weights(w)


def average_weights(model_weights):
    return __apply_on_weights(model_weights, np.average)


def __apply_on_weights(model_weights, np_func):
    weights = []

    # determine how many layers need to be averaged
    n_layers = len(model_weights[0])
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = np.array([m_weight[layer] for m_weight in model_weights])
        # weighted average of weights for this layer
        avg_layer_weights = np_func(layer_weights, axis=0)

        weights.append(avg_layer_weights)

    return weights


def choose(self_rank, high):
    """
    choose a dest_rank from range(size) to push to

    """

    dest_rank = self_rank

    while dest_rank == self_rank:
        dest_rank = np.random.randint(low=0, high=high)

    return dest_rank


def draw(p):
    """
    draw from Bernoulli distribution

    """
    # Bernoulli distribution is a special case of binomial distribution with n=1
    a_draw = np.random.binomial(n=1, p=p, size=None)

    success = (a_draw == 1)

    return success


@tf.function
def kl_loss_compute(logits1, logits2):
    """ KL loss
    """
    pred1 = tf.math.softmax(logits1)
    pred2 = tf.math.softmax(logits2)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(pred2 * tf.math.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss


"""
def average_weights(weights1, weights2):
    weights = []

    for y, w1 in enumerate(weights1):
        w2 = weights2[y]

        w = (w1 + w2) / 2.0

        weights.append(w)

    return weights



def avg_weights(models):
    weights = []

    for m in models:
        w = m.get_weights()

        for i, w1 in enumerate(w):
            if len(weights) == i:
                weights.append(w1)
            else:
                weights[i] += w1

    for wi in range(len(weights)):
        weights[wi] /= len(models)

    return weights

# create a model from the weights of multiple models
def model_weight_ensemble(members, weights):
    # determine how many layers need to be averaged
    n_layers = len(members[0].get_weights())
    # create an set of average model weights
    avg_model_weights = list()
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = array([model.get_weights()[layer] for model in members])
        # weighted average of weights for this layer
        avg_layer_weights = average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)
    # create a new model with the same structure
    model = clone_model(members[0])
    # set the weights in the new
    model.set_weights(avg_model_weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
"""
