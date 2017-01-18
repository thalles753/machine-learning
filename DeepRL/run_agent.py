# -*- coding: utf-8 -*-

import tensorflow as tf
import gym
import os
import argparse
from A3C_Network import A3C_Network
import multiprocessing
from Worker import Worker
import threading
from constants import GLOBAL_NETWORK_NAME
from threading import Lock
import numpy as np
import random

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--game_name", default="DemonAttack-v0", help="Atari game name to be used.")
envarg.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode to run the agent.')
envarg.add_argument('--render', type=bool, default=False, help='Should show the game images.')
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--agent_history_length", type=int, default=4, help="The number of most recent frames experienced by the agent that are given as input to the Q network.")

netarg = parser.add_argument_group('A3C network')
netarg.add_argument("--epsilon", type=float, default=1e-1, help="Epsilon value for RmsProp optimizer.")
netarg.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor gamma used in the Q Learning updates.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.99, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--min_reward", type=float, default=-1.0, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1.0, help="Maximum reward.")
netarg.add_argument("--entropy_regularization", type=float, default=0.01, help="Maximum reward.")

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--tmax", type=int, default=5, help="Perform training after this many game steps.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--epoch_size", type=int, default=4000000, help="How many training steps per epoch.")
mainarg.add_argument("--total_epochs", type=int, default=50, help="How many epochs to run.")
mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")

mainarg = parser.add_argument_group('Debugging variables')
mainarg.add_argument("--average_reward_stats_per_game", type=int, default=1, help="Show learning statistics after this number of epoch.")
mainarg.add_argument("--update_tf_board", type=int, default=10, help="Update the Tensorboard every X steps.")

def sample_learning_rate():
    return np.exp(random.uniform(np.log(10**-4), np.log(10**-2)))

args = parser.parse_args()

tf.reset_default_graph()

model_path = os.path.join('./model/', args.game_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# create openai game
env = gym.make(args.game_name)

n_actions = env.action_space.n

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    # create global network

    main_lock = None

    if args.optimizer == 'rmsprop':
        print "Optimizer algorithm:", args.optimizer
        lr = sample_learning_rate()
        print "Learning rate:", lr
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=args.decay_rate, momentum=0.0, epsilon=args.epsilon)
    elif args.optimizer == 'adam':
        print "Optimizer algorithm:", args.optimizer
        trainer = tf.train.AdamOptimizer(1e-4)

    global_network = A3C_Network(args, n_actions, trainer=trainer, scope=GLOBAL_NETWORK_NAME) # Generate global network

    # get the number of available threads
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for thread_id in range(num_workers):
        workers.append(Worker(args, thread_id, model_path, global_episodes, global_network, trainer, main_lock))

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord)
        t = threading.Thread(target=worker_work)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
