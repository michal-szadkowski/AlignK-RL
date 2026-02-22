from typing import Sequence

import math
import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, obs_shape: Sequence[int], act_size: int, device: torch.device):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device)
        self.value_head = nn.Sequential(
            nn.Linear(math.prod(obs_shape), 1),
        ).to(device)
        self.policy_head = nn.Sequential(
            nn.Linear(math.prod(obs_shape), act_size),
        ).to(device)

    def forward_logits_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.network(x)
        return self.policy_head(y), self.value_head(y)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.network(x))

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.network(x))
