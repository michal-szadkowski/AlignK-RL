import copy

import numpy as np
import torch

from agent import Agent


class OpponentPool:

    def __init__(self, max_size: int, device: torch.device):
        self.max_size = max_size
        self.models = []
        self.device = device

    def add(self, model: Agent):
        snapshot = copy.deepcopy(model).to(self.device).eval()
        self.models.append(snapshot)
        if len(self.models) > self.max_size:
            self.models.pop(0)

    def sample_id(self, envs: int, device: torch.device):
        return torch.randint(0, self.max_size, (envs,), device=device)

    def get_model(self, id: int) -> Agent:
        return self.models[id % len(self.models)]
