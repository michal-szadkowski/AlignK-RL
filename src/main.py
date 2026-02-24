import torch
from torch.distributions import Categorical

from play_with_human import PlayWithHuman
from ppo.agent_module import ConvActorCritic
from env.game import Game
from env.opponent_pool import OpponentPool
from env.selfplay_runner import SelfplayRunner
from ppo.learner import PPOLearner
from ppo.replay_buffer import RolloutBuffer
import time

device = torch.device("cuda")
board_size = 9
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

    if iteration % 5 == 0:
        opponents.add(agent)

    print(f"iter={iteration:4d} run={t1 - t0:.3f}s learn={t2 - t1:.3f}s total={t2 - t0:.3f}s")


env = PlayWithHuman(board_size, k)
human_first = False
agent.to("cpu")
while True:
    env.run(agent, human_first)
    human_first = not human_first
