import torch
from torch.distributions import Categorical

from ppo.agent_module import ConvActorCritic
from env.game import Game
from env.opponent_pool import OpponentPool
from env.selfplay_runner import SelfplayRunner
from ppo.learner import PPOLearner
from ppo.replay_buffer import RolloutBuffer
import time

device = torch.device("cuda")
board_size = 12
k = 5
n_envs = 64

game = Game(board_size, k, n_envs, device)

buffer = RolloutBuffer(256, n_envs, (2, board_size, board_size), (board_size**2,), device)

agent = ConvActorCritic((board_size, board_size), board_size**2, device)

opponents = OpponentPool(agent, 5, device)
runner = SelfplayRunner(game, buffer, opponents)

learner = PPOLearner(buffer, agent)

for iteration in range(500):
    t0 = time.perf_counter()
    runner.run(agent)
    t1 = time.perf_counter()
    learner.learn()
    t2 = time.perf_counter()

    if iteration % 3 == 0:
        opponents.add(agent)

    print(f"iter={iteration:4d} run={t1 - t0:.3f}s learn={t2 - t1:.3f}s total={t2 - t0:.3f}s")


game = Game(board_size, k, 1, device)
while True:
    obs = game.get_canonical_state()

    logits, value = agent.forward(obs)
    act_mask = game.get_legal_moves_mask()
    logits[~act_mask] = -torch.inf
    dist = Categorical(logits=logits)
    action = dist.sample()

    win, draw = game.make_move(action)

    done = win | draw
    if done:
        break

    game.print_nice()
    x = int(input("Move: ").strip())
    y = int(input("Move: ").strip())
    win, draw = game.make_move(torch.tensor([x * board_size + y], device=device))
    done = win | draw
    if done:
        print("Won")
        break
