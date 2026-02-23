import torch
from torch.distributions import Categorical, Distribution

from ppo.actor_critic_module import ActorCriticModule


def sample_action(model: ActorCriticModule, obs, act_mask) -> torch.Tensor:
    logits, _ = model(obs)
    logits[~act_mask] = -torch.inf
    return Categorical(logits=logits).sample()


def get_distribution(model: ActorCriticModule, obs, act_mask) -> tuple[Distribution, torch.Tensor]:
    logits, value = model(obs)
    logits[~act_mask] = -torch.inf
    return Categorical(logits=logits), value
