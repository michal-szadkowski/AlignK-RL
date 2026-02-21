import numpy as np

from agent import Agent


class OpponentPool:

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.models = []

    def add(self, model: Agent):
        self.models.append(model)
        if len(self.models) > self.max_size:
            self.models.pop(0)

    def get_model(self) -> Agent:
        return np.random.choice(self.models)
