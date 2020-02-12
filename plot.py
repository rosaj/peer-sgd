import matplotlib.pyplot as plt
import os


def finalize(ax, title=None):
    # ax.legend(["Train", "Train-validation", "Global validation"], loc=0)  # , "Test - normal data (kontrolno)"
    # ax.set_title("Training/Loss Accuracy per Epoch " + title)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy/Loss")

    path = ''
    if os.getcwd().__contains__('peer-sgd'):
        path = '/Users/robert/Desktop/'
    if title is not None:
        plt.title(title)
        plt.savefig(path + title.replace(".", "_").replace(",", "_").replace(":", "-"))
    if path == '':
        plt.show()


def plot_accuracy_s(historys, title=None, show_legend=True):
    f, ax = plt.subplots()

    for k, v in historys.items():
        ax.plot(v)

    if show_legend:
        ax.legend(list(historys.keys()), loc=0)
    """    
    if 'pm' in historys:
        ax.plot(historys["pm"], color="blue")
    if 'tf' in historys:
        ax.plot(historys["tf"], color="red")
    """
    finalize(ax, title)


def plot_accuracy(history, title=None):
    if hasattr(history, "history"):
        history = history.history

    f, ax = plt.subplots()

    acc_name = "accuracy"
    if acc_name not in history:
        acc_name = "acc"
    ax.plot(history[acc_name], color="blue")

    if "max" in history:
        ax.plot(history["max"], color="black")

    if "test_acc" in history:
        ax.plot(history["test_acc"], color="red")
        ax.legend(["acc", "test"], loc=0)

    finalize(ax, title)

    # f, ax = plt.subplots()
    # ax.plot(history["online"], color="red")
    # finalize(ax, "online_" + title)


"""
    if title is not None:
        ax.set_title(title + " [E: {0}, T_A: {1:.2f}%, T_V_A: {2:.2f}%, G_V_A: {3:.2f}%]".format(n, t_final * 100.,
                                                                                                 t_v_final * 100,
                                                                                                 g_v_final * 100))
        img_name = title.replace(" ", "_") + "_accuracy"
        plt.title(img_name)
        plt.savefig(img_name)
"""
