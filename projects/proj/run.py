import agent
import gym

GAME_NAME= "DemonAttack-v0" # "MsPacman-v0"
env = gym.make(GAME_NAME)

agent = agent.LearningAgent(env, is_training=True)

TOTAL_TRAIN_STEPS = 35000000 # article trained with 10.000.000 million steps

total_reward = 0
EPOCH_SIZE = 10
game_number = 0
for step in range(0,TOTAL_TRAIN_STEPS):

    done, reward = agent.update(step)
    total_reward += reward

    if done:
        game_number += 1

        if game_number % EPOCH_SIZE == 0:
            # write log
            with open('average_score_per_epoch.txt', 'a+') as f:
                f.write(str(game_number) + "\t" + str(total_reward / EPOCH_SIZE) + "\n")

            total_reward = 0
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