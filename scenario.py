from util import *


def interact(sim):
    for d, a in enumerate(sim.agents):
        a.train()


def random_learning(sim):
    num_of_learners = choose(-1, len(sim.agents))

    indices = np.random.choice(len(sim.agents), size=num_of_learners, replace=False)

    for a in np.array(sim.agents)[indices]:
        a.train()


def deterministic_rl(sim):
    for i, a in enumerate(sim.agents):
        if sim.iter % (i + 1) == 0:
            a.train()


def random_active(sim):
    num_of_learners = choose(-1, len(sim.agents))

    indices = np.random.choice(len(sim.agents), size=num_of_learners, replace=False)

    for a in np.array(sim.agents)[indices]:
        if draw(0.02):
            a.active = not a.active

    for a in sim.agents:
        a.train()


def faulty_agents(sim):
    for i in range(len(sim.agents)):
        sim.agents[i].train()

    weights = sim.def_weights
    for i in range(int(len(sim.agents) / 2)):
        a = sim.agents[i]
        a.model.set_weights(weights)
        a.alpha = 2


def data_learner(sim):
    max_t_len = max([len(a.dataset.train) for a in sim.agents])
    for a in sim.agents:
        t_len = len(a.dataset.train)
        if draw(t_len / max_t_len):
            a.train()


def data_learner_r(sim):
    min_t_len = min([len(a.dataset.train) for a in sim.agents])
    max_t_len = max([len(a.dataset.train) for a in sim.agents])
    for a in sim.agents:
        t_len = len(a.dataset.train)
        if draw((max_t_len - t_len + min_t_len) / max_t_len):
            a.train()
