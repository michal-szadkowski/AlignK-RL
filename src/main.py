import torch
from torch.distributions import Categorical

from agent import Agent
from env.game import Game
from env.opponent_pool import OpponentPool
from env.selfplay_runner import SelfplayRunner
from ppo.learner import PPOLearner
from ppo.replay_buffer import RolloutBuffer

device = torch.device("cuda")
board_size = 9
k = 5
n_envs = 64

game = Game(board_size, k, n_envs, device)

buffer = RolloutBuffer(256, n_envs, (2, board_size, board_size), (board_size**2,), device)

opponents = OpponentPool(5, device)

agent = Agent((board_size, board_size), board_size**2, device)
opponents.add(Agent((board_size, board_size), board_size**2, device))

runner = SelfplayRunner(game, buffer, opponents)

learner = PPOLearner(buffer, agent)

for iteration in range(500):
    runner.run(agent)
    learner.learn()
    if iteration % 3 == 0:
        opponents.add(agent)
