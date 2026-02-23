import copy
import torch
from ppo.actor_critic_module import ActorCriticModule


class OpponentPool:

    def __init__(self, current_agent, max_size: int, device: torch.device):
        self.max_size = max_size
        self.models = []
        self.device = device
        self.current_agent = current_agent

    def add(self, model: ActorCriticModule):
        snapshot = copy.deepcopy(model).to(self.device).eval()
        self.models.append(snapshot)
        if len(self.models) > self.max_size:
            self.models.pop(0)

    def sample_id(self, envs: int, device: torch.device):
        n = len(self.models)
        return torch.randint(0, n + 1, (envs,), device=device)

    def get_model(self, id: int) -> ActorCriticModule:
        n = len(self.models)
        if n == 0 or id == n:
            return self.current_agent
        return self.models[id]
