import agent
import gym

GAME_NAME= "DemonAttack-v0" # "MsPacman-v0"
env = gym.make(GAME_NAME)

agent = agent.LearningAgent(env, is_training=True)

TOTAL_TRAIN_STEPS = 35000000 # article trained with 10.000.000 million steps

for step in range(0,TOTAL_TRAIN_STEPS):

    done = agent.update(step)

    if done:
        print("Episode finished after {} timesteps".format(step))


# 0 - 'NOOP'
# 1 - 'UP'
# 2 - 'RIGHT'
# 3 - 'LEFT'
# 4 - 'DOWN'
# 5 - 'UPRIGHT'
# 6 - 'UPLEFT'
# 7 - 'DOWNRIGHT'
# 8 - 'DOWNLEFT']