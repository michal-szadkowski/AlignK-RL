import torch
import torch.nn.functional as F


class Game:
    def __init__(self, board_size: int, win_condition: int, device: torch.device):
        self.board_size = board_size
        self.win_condition = win_condition
        self.max_moves = board_size * board_size

        self.device = device

        self.filters = self._create_win_filters().to(self.device)
        self.reset()

    def reset(self):
        self.board = torch.zeros(2, self.board_size, self.board_size, dtype=torch.float32, device=self.device)
        self.current_player = torch.zeros((1,), dtype=torch.int)
        self.move_count = torch.zeros((1,), dtype=torch.int)

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
        if self.current_player == 1:
            state = torch.flip(self.board, dims=[0])
        else:
            state = self.board.clone()

        return state.unsqueeze(0)

    def make_move(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = torch.unravel_index(action, (self.board_size, self.board_size))

        if self.board[:, x, y].sum() > 0:
            raise ValueError(f"Space ({x}, {y}) is taken")

        self.board[self.current_player, x, y] = 1.0
        self.move_count += 1

        win = self.check_win_full()
        draw = self.check_draw()

        if not (win | draw):
            self.current_player = 1 - self.current_player

        return win, draw

    def check_win_full(self) -> torch.Tensor:
        n = self.win_condition
        layer = self.board[self.current_player]

        results = F.conv2d(layer, self.filters, padding=n // 2)

        return (results >= n).any()

    def get_legal_moves_mask(self) -> torch.Tensor:
        return (self.board.flatten(-2, -1).sum(dim=0) == 0).unsqueeze(0)

    def check_draw(self) -> torch.Tensor:
        return (self.move_count >= self.max_moves).to(self.device)
