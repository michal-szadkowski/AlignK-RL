import torch
from env.player import get_distribution, sample_action
from ppo.actor_critic_module import ActorCriticModule
from env.game import Game
from env.opponent_pool import OpponentPool
from ppo.replay_buffer import RolloutBuffer


class SelfplayRunner:
    def __init__(self, game: Game, buffer: RolloutBuffer, opponent_pool: OpponentPool):
        self.game = game
        self.n_envs = game.n_envs
        self.buffer = buffer
        self.opponent_pool = opponent_pool

        self.opponents = opponent_pool.sample_id(self.n_envs, self.buffer.device)

        self.play_as = torch.zeros(self.n_envs, dtype=torch.int8, device=self.buffer.device)

        self.ep_lengths = torch.zeros(self.n_envs, dtype=torch.int, device=self.buffer.device)
        self.reset()

    def run(self, player: ActorCriticModule):
        with torch.no_grad():
            avg_rewards = []
            avg_ep_len = []

            for i in range(self.buffer.n_steps):

                obs = self.game.get_canonical_state()
                act_mask = self.game.get_legal_moves_mask()

                dist, value = get_distribution(player, obs, act_mask)
                action = dist.sample()

                rewards = torch.zeros((self.n_envs,), device=self.buffer.device)

                win, draw = self.game.make_move(action)
                done_after_player = win | draw
                rewards[win] = 1

                lose, draw2 = self._make_opponent_move(~done_after_player)
                done_after_opponent = lose | draw2
                rewards[lose] = -1

                self.buffer.add(
                    obs,
                    action.unsqueeze(-1),
                    rewards,
                    value.squeeze(-1),
                    dist.log_prob(action),
                    done_after_player | done_after_opponent,
                    act_mask,
                )

                episode_done = done_after_player | done_after_opponent

                self.ep_lengths += 1
                if episode_done.any():
                    avg_rewards.append(rewards[episode_done].mean().cpu().item())
                    finished_lengths = self.ep_lengths[episode_done].detach().cpu().tolist()
                    avg_ep_len.extend(finished_lengths)

                self.reset(episode_done)

            _, last_val = get_distribution(player, self.game.get_canonical_state(), self.game.get_legal_moves_mask())
            self.buffer.compute_advantages(last_val.squeeze(-1))

            print(sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0.0)
            print(sum(avg_ep_len) / len(avg_ep_len) if avg_ep_len else 0.0)

    def _make_opponent_move(self, envs_to_move: torch.Tensor):
        if not envs_to_move.any():
            return torch.zeros_like(envs_to_move), torch.zeros_like(envs_to_move)

        obs_all = self.game.get_canonical_state()
        mask_all = self.game.get_legal_moves_mask()

        active_idx = torch.where(envs_to_move)[0]

        opp_ids_active = self.opponents[active_idx]
        action_active = torch.empty(active_idx.numel(), dtype=torch.long, device=obs_all.device)

        for opp_id in opp_ids_active.unique():
            grp = opp_ids_active == opp_id
            grp_idx = active_idx[grp]

            model = self.opponent_pool.get_model(int(opp_id.item()))
            action = sample_action(model, obs_all[grp_idx], mask_all[grp_idx])
            action_active[grp] = action

        win, draw = self.game.make_move(action_active, envs_to_move)
        return win, draw

    def reset(self, envs_to_reset: torch.Tensor | None = None):
        if envs_to_reset is None:
            envs_to_reset = torch.ones(self.n_envs, dtype=torch.bool, device=self.buffer.device)

        if envs_to_reset.any():
            self.game.reset(envs_to_reset)
            self._resample_play_as(envs_to_reset)
            self._make_first_move(envs_to_reset)
            self.opponents[envs_to_reset] = self.opponent_pool.sample_id(
                int(envs_to_reset.sum().item()), self.buffer.device
            )
            self.ep_lengths[envs_to_reset] = 0

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
