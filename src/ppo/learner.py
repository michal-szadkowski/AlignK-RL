from torch.distributions import Categorical
import torch.optim
import torch.nn.functional as F
from agent import Agent
from ppo.replay_buffer import RolloutBuffer


class PPOLearner:
    def __init__(self, buffer: RolloutBuffer, agent: Agent):
        self.buffer = buffer
        self.agent = agent
        self.optimizer = torch.optim.AdamW(agent.parameters())
        self.minibatch_size = 512
        self.epochs = 10
        self.clip_coef = 0.2

        self.critic_loss_coef = 0.5

    def learn(self):
        for epoch in range(self.epochs):
            for batch in self.buffer.get_enumerator(self.minibatch_size):
                self.optimizer.zero_grad()

                new_logits, new_value = self.agent.forward_logits_value(batch.observations)
                new_logits[~batch.act_mask] = -torch.inf
                dist = Categorical(logits=new_logits)

                new_log_probs = dist.log_prob(batch.actions)

                ratio = torch.exp(new_log_probs - batch.log_probs)

                pg_loss1 = -batch.advantages * ratio
                pg_loss2 = -batch.advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                critic_loss = F.mse_loss(new_value.squeeze(-1), batch.returns)

                total_loss = actor_loss.float() + self.critic_loss_coef * critic_loss.float()

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.reset()
