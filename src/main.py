import torch

from agent import Agent
from env.game import Game
from env.opponent_pool import OpponentPool
from env.selfplay_runner import SelfplayRunner
from ppo.learner import PPOLearner
from ppo.replay_buffer import RolloutBuffer

device = torch.device("cuda")
side = 6
game = Game(side, 3, device)
buffer = RolloutBuffer(512, 1, (2, side, side), (side * side,), device)

opponents = OpponentPool(5)

agent = Agent((side, side), side * side, device)
opponents.add(Agent((side, side), side * side, device))

runner = SelfplayRunner(game, buffer, opponents)

learner = PPOLearner(buffer, agent)

for iteration in range(1000):
    runner.run(agent)
    learner.learn()

print("h")
