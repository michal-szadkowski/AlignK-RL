import torch

from env.game import Game
from env.player import sample_action
from ppo.actor_critic_module import ActorCriticModule


class PlayWithHuman:
    def __init__(self, board_size, win_cond):
        self.board_size = board_size
        self.k = win_cond

    def run(self, module: ActorCriticModule, human_first=False):
        game = Game(self.board_size, self.k, 1, torch.device("cpu"))
        human_turn = human_first
        while True:
            if human_turn:
                self._make_human_move(game)
            else:
                self._make_model_move(game, module)

            human_turn = not human_turn

    def _make_model_move(self, game, module):
        obs = game.get_canonical_state()
        act_mask = game.get_legal_moves_mask()
        action = sample_action(module, obs, act_mask)
        win, draw = game.make_move(action)

        done = win | draw
        if done:
            game.print_nice()
            if win.any():
                print("Model won.")
            else:
                print("Draw.")
            return True
        return False

    def _make_human_move(self, game):
        game.print_nice()

        human_act = self._get_human_action()
        win, draw = game.make_move(human_act)
        done = win | draw
        if done:
            game.print_nice()
            if win.any():
                print("Human won.")
            else:
                print("Draw.")
            return True
        return False

    def _get_human_action(self):
        row = int(input("Row: ").strip())
        col = int(input("Col: ").strip())
        return torch.tensor([row * self.board_size + col])
