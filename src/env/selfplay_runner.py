import torch
from agent import Agent
from env.game import Game
from env.opponent_pool import OpponentPool
from ppo.replay_buffer import RolloutBuffer

from torch.distributions import Categorical


class SelfplayRunner:
    def __init__(self, game: Game, buffer: RolloutBuffer, opponent_pool: OpponentPool):
        self.game = game
        self.buffer = buffer
        self.opponent_pool = opponent_pool
        self.opponent = opponent_pool.get_model()

    def run(self, player: Agent):
        with torch.no_grad():
            avg_rewards = []
            for i in range(self.buffer.n_steps):

                obs = self.game.get_canonical_state()

                logits, value = player.forward_logits_value(obs)
                act_mask = self.game.get_legal_moves_mask()
                logits[~act_mask] = -torch.inf
                dist = Categorical(logits=logits)
                action = dist.sample()

                rewards = torch.zeros((1,), device=self.buffer.device)

                win, draw = self.game.make_move(action)

                done = win | draw
                rewards[win] = 1

                lose, draw2 = self.make_opponent_move(~done)

                done_after_opponent = lose | draw2
                rewards[lose] = -1

                self.buffer.add(
                    obs,
                    action.unsqueeze(0),
                    rewards,
                    value.squeeze(),
                    dist.log_prob(action),
                    done | done_after_opponent,
                    act_mask,
                )
                if done | done_after_opponent:
                    avg_rewards.append(rewards.mean().cpu().item())

                self.reset(done | done_after_opponent)

            last_val = player.forward_value(self.game.get_canonical_state())
            self.buffer.compute_advantages(last_val)
            print(sum(avg_rewards) / len(avg_rewards))

    def make_opponent_move(self, envs_to_move: torch.Tensor):
        if ~envs_to_move.any():
            return torch.zeros_like(envs_to_move), torch.zeros_like(envs_to_move)

        obs = self.game.get_canonical_state()

        logits = self.opponent.forward_logits(obs)
        act_mask = self.game.get_legal_moves_mask()
        logits[~act_mask] = -torch.inf
        dist = Categorical(logits=logits)
        action = dist.sample()

        win, draw = self.game.make_move(action)
        return win, draw

    def reset(self, envs_to_reset: torch.Tensor):
        if envs_to_reset.any():
            self.game.reset()
