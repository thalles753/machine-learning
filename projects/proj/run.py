import agent
import gym
import argparse

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--game_name", default="DemonAttack-v0", help="Atari game name to be used.")
envarg.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode to run the agent.')
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_memory_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--agent_history_length", type=int, default=4, help="The number of most recent frames experienced by the agent that are given as input to the Q network.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor gamma used in the Q Learning updates.")
netarg.add_argument("--minibatch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--gradient_momentum", type=float, default=0.95, help="Gradient momentum used in RMSProp")
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--initial_exploration", type=float, default=1, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--final_exploration", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--final_exploration_frame", type=float, default=1000000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--initial_exploration_test", type=float, default=0.05, help="Exploration rate used during testing.")
antarg.add_argument("--update_frequency", type=int, default=4, help="Perform training after this many game steps.")
antarg.add_argument("--target_network_update_frequency", type=int, default=10000, help="The frequency (measured in the number of SGD updates) with which the target network is updated.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--replay_start_size", type=int, default=5000, help="Populate replay memory with random steps before starting learning.")
mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
mainarg.add_argument("--debug_epsode_size", type=int, default=20, help="How many games before calculating probabilities.")

args = parser.parse_args()
env = gym.make(args.game_name)

agent = agent.LearningAgent(env, args)

agent.play()

# step = 0
# game_number = 0
# EPOCH_SIZE = 10
# total_reward = 0
# TOTAL_STEPS = args.epochs * args.train_steps
#
# for step in range(50000000):
#
#     done, reward = agent.update(step)
#     total_reward += reward
#
#     if done:
#         game_number += 1
#         agent.update(step)
#
#         if game_number % EPOCH_SIZE == 0:
#             # write log
#             with open('average_score_per_epoch.txt', 'a+') as f:
#                 f.write(str(game_number) + "\t" + str(total_reward / EPOCH_SIZE) + "\n")
#             total_reward = 0