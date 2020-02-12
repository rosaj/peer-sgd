from collections import OrderedDict


class ModelHistory:

    def __init__(self, metrics):
        self.history = OrderedDict({m: [] for m in metrics})

    def update(self, values):
        for i, metric in enumerate(self.history):
            self.history[metric].append(values[i])

    def print(self):
        for key, val in self.history.items():
            print(key, val)
