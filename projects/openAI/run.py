import gym
import pdb
import agent as ag
from time import sleep

env = gym.make('MsPacman-v0')
agent = ag.LearningAgent(env)

DELAY = 0.1
TOTAL_EPSODES = 20
EPSODE_SIZE = 100

for i_episode in range(TOTAL_EPSODES):
    for t in range(EPSODE_SIZE):
        step = i_episode * EPSODE_SIZE + t

        agent.update(step)

        sleep(DELAY)
