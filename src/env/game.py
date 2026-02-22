import torch
import torch.nn.functional as F


class Game:
    def __init__(self, board_size: int, win_condition: int, n_envs: int, device: torch.device):
        self.board_size = board_size
        self.win_condition = win_condition
        self.n_envs = n_envs
        self.max_moves = board_size * board_size

        self.device = device

        self.filters = self._create_win_filters().to(self.device)
        self.board = torch.zeros(
            self.n_envs,
            2,
            self.board_size,
            self.board_size,
            dtype=torch.float32,
            device=self.device,
        )
        self.current_player = torch.zeros(self.n_envs, dtype=torch.int, device=device)
        self.move_count = torch.zeros(self.n_envs, dtype=torch.int, device=device)

    def reset(self, envs_to_reset=None):
        if envs_to_reset is None:
            self.board.zero_()
            self.current_player.zero_()
            self.move_count.zero_()
            return

        if not envs_to_reset.any():
            return

        self.board[envs_to_reset] = 0.0
        self.current_player[envs_to_reset] = 0
        self.move_count[envs_to_reset] = 0

    def _create_win_filters(self) -> torch.Tensor:
        n = self.win_condition
        f = torch.zeros((4, 1, n, n), dtype=torch.float32)

        for i in range(n):
            f[0, 0, n // 2, i] = 1.0
            f[1, 0, i, n // 2] = 1.0
            f[2, 0, i, i] = 1.0
            f[3, 0, i, n - 1 - i] = 1.0
        return f

    def get_canonical_state(self):
        env_idx = torch.arange(self.n_envs, device=self.device)
        chan_idx = torch.stack([self.current_player, 1 - self.current_player], dim=1)

        return self.board[env_idx[:, None], chan_idx]

    def make_move(
        self, action: torch.Tensor, envs_to_move: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if envs_to_move is None:
            envs_to_move = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)

        if action.numel() != int(envs_to_move.sum().item()):
            raise ValueError("action must match number of active envs")

        if not envs_to_move.any():
            zeros = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            return zeros, zeros

        row = action // self.board_size
        col = action % self.board_size

        env_idx = torch.arange(self.n_envs, device=self.device)[envs_to_move]
        cp = self.current_player[envs_to_move]

        occupied = self.board[env_idx, 0, row, col] + self.board[env_idx, 1, row, col] > 0
        if occupied.any():
            raise ValueError("Some selected moves are occupied.")

        self.board[env_idx, cp, row, col] = 1.0
        self.move_count[envs_to_move] += 1

        win = self.check_win_full() & envs_to_move
        draw = self.check_draw() & envs_to_move
        done = win | draw

        self.current_player[envs_to_move] = torch.where(done[envs_to_move], cp, 1 - cp)

        return win, draw

    def check_win_full(self) -> torch.Tensor:
        n = self.win_condition
        env_idx = torch.arange(self.n_envs, device=self.device)

        layer = self.board[env_idx, self.current_player]

        results = F.conv2d(layer.unsqueeze(1), self.filters, padding=n // 2)

        return (results >= n).any(dim=(1, 2, 3))

    def get_legal_moves_mask(self) -> torch.Tensor:
        occupied = self.board.flatten(2).sum(dim=1) > 0
        return ~occupied

    def check_draw(self) -> torch.Tensor:
        return self.move_count >= self.max_moves

    def print_nice(self):
        size = self.board_size
        idx_width = max(2, len(str(size - 1)))
        cell_width = max(2, idx_width)

        for env in range(self.n_envs):
            print(f"\nEnv {env}")
            print(" " * (idx_width + 3), end="")
            for col in range(size):
                print(f"{col:>{cell_width}}", end=" ")
            print()

            separator = " " * (idx_width + 1) + "+" + "-" * ((cell_width + 1) * size + 1)
            print(separator)

            for row in range(size):
                print(f"{row:>{idx_width}} |", end=" ")
                for col in range(size):
                    if self.board[env, 0, row, col] == 1:
                        symbol = "O"
                    elif self.board[env, 1, row, col] == 1:
                        symbol = "X"
                    else:
                        symbol = "."
                    print(f"{symbol:>{cell_width}}", end=" ")
                print()
            print()
