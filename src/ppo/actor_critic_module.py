from typing import NamedTuple
import torch
from torch import nn
from abc import ABC, abstractmethod


class ActorCriticOut(NamedTuple):
    logits: torch.Tensor
    value: torch.Tensor


class ActorCriticModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> ActorCriticOut:
        pass
