from typing import NamedTuple

import torch


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, act_shape, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.n_steps, self.n_envs, *self.obs_shape).to(self.device)

        self.actions = torch.zeros(self.n_steps, self.n_envs, len(self.act_shape)).to(self.device)
        self.action_masks = torch.zeros(self.n_steps, self.n_envs, *self.act_shape, dtype=torch.bool).to(self.device)
        self.log_probs = torch.zeros(self.n_steps, self.n_envs).to(self.device)

        self.rewards = torch.zeros(self.n_steps, self.n_envs).to(self.device)
        self.dones = torch.zeros(self.n_steps, self.n_envs).to(self.device)
        self.values = torch.zeros(self.n_steps, self.n_envs).to(self.device)

        self.advantages = torch.zeros(self.n_steps, self.n_envs).to(self.device)
        self.returns = torch.zeros(self.n_steps, self.n_envs).to(self.device)

        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done, action_mask):
        if self.ptr >= self.n_steps:
            raise IndexError("Buffer full")

        self.obs[self.ptr].copy_(obs)

        self.actions[self.ptr].copy_(action)
        self.action_masks[self.ptr].copy_(action_mask)
        self.log_probs[self.ptr].copy_(log_prob)

        self.rewards[self.ptr].copy_(reward)
        self.dones[self.ptr].copy_(done)
        self.values[self.ptr].copy_(value)

        self.ptr += 1

    def compute_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
        with torch.no_grad():
            last_gae = torch.zeros(self.n_envs).to(self.device)
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextvalues = last_values
                else:
                    nextvalues = self.values[t + 1]

                next_non_terminal = 1.0 - self.dones[t].float()

                delta = self.rewards[t] + gamma * nextvalues * next_non_terminal - self.values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                self.advantages[t] = last_gae
            self.returns = self.advantages + self.values

    def get_enumerator(self, batch_size):
        steps = self.ptr
        num_samples = steps * self.n_envs

        b_obs = self.obs[:steps].view(num_samples, *self.obs_shape)
        b_actions = self.actions[:steps].view(num_samples)
        b_log_probs = self.log_probs[:steps].view(num_samples)

        b_returns = self.returns[:steps].view(num_samples)
        b_advantages = self.advantages[:steps].view(num_samples)
        b_values = self.values[:steps].view(num_samples)
        b_masks = self.action_masks[:steps].view(num_samples, *self.act_shape)

        indices = torch.randperm(num_samples, device=self.device)

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield BufferBatch(
                b_obs[batch_idx],
                b_actions[batch_idx],
                b_masks[batch_idx],
                b_log_probs[batch_idx],
                b_advantages[batch_idx],
                b_returns[batch_idx],
            )


class BufferBatch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    act_mask: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
