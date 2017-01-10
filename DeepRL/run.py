import tensorflow as tf
import gym
import os
import argparse
from A3C_Network import A3C_Network
import multiprocessing
from Worker import  Worker

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--game_name", default="DemonAttack-v0", help="Atari game name to be used.")
envarg.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode to run the agent.')
envarg.add_argument('--render', type=bool, default=False, help='Should show the game images.')
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")

netarg = parser.add_argument_group('A3C network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor gamma used in the Q Learning updates.")
netarg.add_argument("--minibatch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.99, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--initial_exploration", type=float, default=1.0, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--final_exploration", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--final_exploration_frame", type=float, default=1000000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--update_frequency", type=int, default=4, help="Perform training after this many game steps.")
antarg.add_argument("--target_network_update_frequency", type=int, default=10000, help="The frequency (measured in the number of SGD updates) with which the target network is updated.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")

mainarg = parser.add_argument_group('Debugging variables')
mainarg.add_argument("--average_reward_stats_per_game", type=int, default=10, help="Show learning statistics after this number of epoch.")
mainarg.add_argument("--update_tf_board", type=int, default=1000, help="Update the Tensorboard every X steps.")


args = parser.parse_args()

tf.reset_default_graph()

model_path = os.path.join('./model/', args.game_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

# create openai game
env = gym.make(args.game_name)

input_shape = (args.screen_width, args.screen_height, args.agent_history_length) # 84 x 84 x 4
n_actions = env.action_space.n

# create global network
master_network = A3C_Network(input_shape, n_actions, 'global') # Generate global network

# get the number of available threads
num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads

workers = []
# Create worker classes
for thread_id in range(num_workers):
    workers.append(Worker(env, thread_id, input_shape, n_actions,model_path, global_episodes))


# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     if load_model == True:
#         print 'Loading Model...'
#         ckpt = tf.train.get_checkpoint_state(model_path)
#         saver.restore(sess,ckpt.model_checkpoint_path)
#     else:
#         sess.run(tf.global_variables_initializer())
#
#     # This is where the asynchronous magic happens.
#     # Start the "work" process for each worker in a separate threat.
#     worker_threads = []
#     for worker in workers:
#         worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
#         t = threading.Thread(target=(worker_work))
#         t.start()
#         worker_threads.append(t)
#     coord.join(worker_threads)