import torch
from agent import Agent
from env.game import Game
from env.opponent_pool import OpponentPool
from ppo.replay_buffer import RolloutBuffer

from torch.distributions import Categorical


class SelfplayRunner:
    def __init__(self, game: Game, buffer: RolloutBuffer, opponent_pool: OpponentPool):
        self.game = game
        self.n_envs = game.n_envs
        self.buffer = buffer
        self.opponent_pool = opponent_pool
        self.opponent = opponent_pool.get_model()

        self.play_as = torch.zeros(self.n_envs, dtype=torch.int8, device=self.buffer.device)
        self.reset()

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

                rewards = torch.zeros((self.n_envs,), device=self.buffer.device)

                win, draw = self.game.make_move(action)

                done = win | draw
                rewards[win] = 1

                lose, draw2 = self._make_opponent_move(~done)

                done_after_opponent = lose | draw2
                rewards[lose] = -1

                self.buffer.add(
                    obs,
                    action.unsqueeze(-1),
                    rewards,
                    value.squeeze(-1),
                    dist.log_prob(action),
                    done | done_after_opponent,
                    act_mask,
                )

                if (done | done_after_opponent).any():
                    avg_rewards.append(rewards[done | done_after_opponent].mean().cpu().item())

                self.reset(done | done_after_opponent)

            last_val = player.forward_value(self.game.get_canonical_state())
            self.buffer.compute_advantages(last_val.squeeze(-1))

            print(sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0.0)

    def _make_opponent_move(self, envs_to_move: torch.Tensor):
        if not envs_to_move.any():
            return torch.zeros_like(envs_to_move), torch.zeros_like(envs_to_move)

        obs = self.game.get_canonical_state()[envs_to_move]

        logits = self.opponent.forward_logits(obs)
        act_mask = self.game.get_legal_moves_mask()[envs_to_move]
        logits[~act_mask] = -torch.inf
        dist = Categorical(logits=logits)
        action = dist.sample()

        win, draw = self.game.make_move(action, envs_to_move)
        return win, draw

    def reset(self, envs_to_reset: torch.Tensor | None = None):
        if envs_to_reset is None:
            envs_to_reset = torch.ones(self.n_envs, dtype=torch.bool, device=self.buffer.device)

        if envs_to_reset.any():
            self.game.reset(envs_to_reset)
            self._resample_play_as(envs_to_reset)
            self._make_first_move(envs_to_reset)

    def _resample_play_as(self, envs: torch.Tensor):
        if envs.any():
            self.play_as[envs] = torch.randint_like(self.play_as[envs], 0, 2)

    def _make_first_move(self, envs: torch.Tensor):
        second = envs & (self.play_as == 1)
        if not second.any():
            return

        lose, draw = self._make_opponent_move(second)
        done = lose | draw

        if done.any():
            self.reset(done)
