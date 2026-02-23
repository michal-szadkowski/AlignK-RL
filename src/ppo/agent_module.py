from typing import Sequence
import math
import torch
import torch.nn as nn
from ppo.actor_critic_module import ActorCriticModule, ActorCriticOut


class ConvActorCritic(ActorCriticModule):
    def __init__(self, obs_shape: Sequence[int], act_size: int, device: torch.device):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
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

    def forward(self, x: torch.Tensor) -> ActorCriticOut:
        y = self.network(x)
        return ActorCriticOut(self.policy_head(y), self.value_head(y))
