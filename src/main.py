import torch

from agent import Agent
from env.game import Game
from env.opponent_pool import OpponentPool
from env.selfplay_runner import SelfplayRunner
from ppo.learner import PPOLearner
from ppo.replay_buffer import RolloutBuffer

device = torch.device("cuda")
board_size = 6
n_envs = 16

game = Game(board_size, 3, n_envs, device)

buffer = RolloutBuffer(64, n_envs, (2, board_size, board_size), (board_size**2,), device)

opponents = OpponentPool(5)

agent = Agent((board_size, board_size), board_size**2, device)
opponents.add(Agent((board_size, board_size), board_size**2, device))

runner = SelfplayRunner(game, buffer, opponents)

learner = PPOLearner(buffer, agent)

for iteration in range(1000):
    runner.run(agent)
    learner.learn()

print("h")
