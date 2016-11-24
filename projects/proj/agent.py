# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from collections import deque
from scipy import misc
import DQN
import random
import json
import csv
import utils
import matplotlib.pyplot as plt


class LearningAgent:
    def __init__(self, env, init_exp_prob=1.0, is_training=True):
        # game variables and constants
        self._env = env
        self.state = None
        self.action = None
        self.ACTION_NAMES = self._env.get_action_meanings()
        self.NUM_ACTIONS = env.action_space.n
        self.train_step = 0
        self.rewards_per_game = []
        self.is_training = is_training
        self.IMAGE_SIZE = 84
        self.PIXEL_DEPTH = 255.0
        self.game_number = 0

        # Debug variables
        self.reset_debug_variables()
        self.rewards_distribution = dict([(key, 0) for key in self.ACTION_NAMES])

        # AI variables and constants
        self.REPLAY_MEN_SIZE = 1000000 # The paper remember the 1.000.000 most recent frames
        self.exp_replay_list = deque(maxlen=self.REPLAY_MEN_SIZE) # Initialize replay memory
        self.ACTION_REPEAT = 4 # repeat each actions selected by the agent this many times
        self.UPDATE_FREQUENCY = 4 # number of actions selected by the agent between successive SGD updates
        self.REPLAY_START_SIZE = 500 # Papers value: 50000
        self.AGENT_HISTORY_LENGTH = 4
        self.TARGET_NET_UPDATE_FREQUENCE = 10000
        self.INITIAL_EXPLORATION_PROB = init_exp_prob # Exploration probability
        self.FINAL_EXPLORATION_PROB = 0.1
        self.learning_rate = 0.00025 # lr from the paper 0.00025
        self.discount_factor = 0.99 # dicount factor from the paper 0.99
        self.MINI_BATCH_SIZE = 32 # minibatch size from the paper 32
        self.FINAL_EXPLORATION_FRAME = 1000000 # the number of frames over which the initial value of [INITIAL_EXPLORATION_PROB] is linearly annealed to its final value
        self.SAVE_MODEL_EVERY = 10000
        self.UPDATE_TF_BOARD_EVERY = 1000
        self.ACTION_REPEAT = 4
        self.GRADIENT_MOMENTUM = 0.95

        # Tensorflow variables
        self._session = tf.Session()
        self._target = tf.placeholder(tf.float32, [None], name="input_targets")
        self._action = tf.placeholder(tf.float32, [None, self.NUM_ACTIONS], name="input_actions")

        # setup log file
        with open('log.csv', 'w+') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            fieldnames = ['Global_step', 'Game_number', 'Traning_step', 'Loss', 'Last_action', 'Random_prob','#_of_Random_actions', '#_of_Conscious_actions']
            writer.writerow(fieldnames)

        network_input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, self.AGENT_HISTORY_LENGTH)

        # action-value function Q
        with tf.variable_scope("train") as self.target_scope:
            self.train_network = DQN.Network(network_input_shape, self.NUM_ACTIONS)

        # target action-value function Q_hat
        with tf.variable_scope("target") as self.train_scope:
            self.target_network = DQN.Network(network_input_shape, self.NUM_ACTIONS)

        readout_action = tf.reduce_sum(tf.mul(self.train_network.logits, self._action), reduction_indices=1)

        diff = self._target - readout_action
        diff_clipped = tf.clip_by_value(diff, -1, 1)
        self.loss = tf.reduce_mean(tf.square(diff_clipped))
        loss_summary = tf.scalar_summary('loss', self.loss)

        self.train_operation = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)

        self.merged = tf.merge_summary([loss_summary])
        self.train_writer = tf.train.SummaryWriter('./summary', self._session.graph)

        self._session.run(tf.initialize_all_variables())

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        self.count = 0
        self.game_initial_setup()

    def reset_debug_variables(self):
        self.number_of_random_actions = 0
        self.number_of_systematic_actions = 0

    def to_one_hot_vec(self, actions):
        one_hot = np.zeros((self.MINI_BATCH_SIZE, self.NUM_ACTIONS))
        one_hot[np.arange(self.MINI_BATCH_SIZE), actions] = 1
        return one_hot

    def compare_train_target_net_random_weight(self):
        v1 = self.train_network.get_training_var_data(self._session)
        v2 = self.target_network.get_training_var_data(self._session)

        n = random.randint(0, 32);

        print("TrainNet\tTargetNet")
        print str(v1[0][0][0][n]) + "\t" + str(v2[0][0][0][n])

        if np.array_equal(v1,v2):
            print "Networks Equal!!"
        else:
            print "Networks Not equal..."

    def check_previous_post_state(self, prev, post):
        print np.array_equal(prev[0], post[2])

        #assert((prev.shape != post.shape), "State objects with different shapes")
        #assert((np.array_equal(prev[1], post[1]) == True), "Error checking prev/post states")

    def display_experience_men(self, memory):
        # displya post states
        states = [memory[0], memory[3]]
        id = 0
        f, axarr = plt.subplots(2, 4, figsize=(18,8))
        for state in states:
            for c in range(0,4):
                axarr[id][c].imshow(state[:,:,c], cmap=plt.cm.Greys);
                axarr[id][c].set_title('Action' + str(memory[1]))
            id += 1
        plt.show()

    def train(self):
        # Sample random minibatch of transitions (s_i, a_i, r_i, s_i+1)
        # from the experience replay list
        mini_batch = random.sample(self.exp_replay_list, self.MINI_BATCH_SIZE)
        assert(len(mini_batch) == self.MINI_BATCH_SIZE)

        previous_states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        current_states = [d[3] for d in mini_batch]
        terminals = [d[4] for d in mini_batch]


        # clip rewards between -1 and 1
        rewards = np.clip(rewards, a_min=-1, a_max=1)

        if self.train_step % self.TARGET_NET_UPDATE_FREQUENCE == 0:
            self.compare_train_target_net_random_weight()

            # update target network
            self.target_network.copy_weights(self.train_network, self._session)
            print "Target network updated."

            self.compare_train_target_net_random_weight()



        # this gives us the agents expected reward for each action we might take
        poststate_qvalues = self._session.run(self.target_network.logits, feed_dict={self.target_network._input: current_states})
        max_poststate_qvalues = np.max(poststate_qvalues, 1)

        agents_expected_reward = []
        for i in range(len(mini_batch)):
            if terminals[i]:
                # this was a terminal state so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                # compute r_j + γ max Q(s_j, a; θ),
                #    where:
                #      s_j is the current state
                #      θ are the CNNs learned weights
                agents_expected_reward.append(
                    rewards[i] + self.discount_factor * max_poststate_qvalues[i])

        # Perform a gradient descent step on (y_j − Q(s_j, a_j; θ))^2
        summary, _, loss = self._session.run([self.merged, self.train_operation, self.loss], feed_dict={
            self.train_network._input: previous_states,
            self._action: self.to_one_hot_vec(actions),
            self._target: agents_expected_reward})


        if self.train_step % self.UPDATE_TF_BOARD_EVERY == 0:
            # print "Summary data has been written!!"
            self.train_writer.add_summary(summary, self.train_step)

        self.train_step += 1

        return loss

    def reset_env(self):
        obs = self._env.reset()
        self.lives = self._env.ale.lives()
        return obs

    def get_random_action(self):
        return self._env.action_space.sample()

    def get_action_name(self, action):
        return self.ACTION_NAMES[action]

    # for the first step, the state is the same state repeated [STATE_FRAMES] times
    def game_initial_setup(self):
        first_frame = self.reset_env()
        processed_frame = self.process_input(first_frame)
        self.state = np.stack(tuple(processed_frame for _ in range(self.AGENT_HISTORY_LENGTH)), axis=2)
        assert(self.state.shape == (self.IMAGE_SIZE,self.IMAGE_SIZE,self.AGENT_HISTORY_LENGTH))

        for i in range(0,4):
            for j in range(0,4):
                if np.array_equal(self.state[i], self.state[j]) == False:
                    exit("Not equal")

        self._env.frameskip = self.ACTION_REPEAT
        print "Game setup and ready to go!"

    def get_next_action(self, step):
        # With probability p select a random action (a) otherwise select a = maxaQ∗(φ(st), a; θ)
        p = self.get_exploration_probability(step)

        if random.random() < p:
            out = self.get_random_action()
            self.number_of_random_actions += 1
        else:
            current_state = np.expand_dims(self.state, axis=0)
            q_action = self._session.run(self.train_network.logits, feed_dict={self.train_network._input: current_state})
            out = np.argmax(q_action)
            self.number_of_systematic_actions += 1

        self.rewards_distribution[self.get_action_name(out)] += 1
        return out, p

    # perform a linear decay operation
    def get_exploration_probability(self, global_step):
        return (self.FINAL_EXPLORATION_PROB +
                max(0, (self.INITIAL_EXPLORATION_PROB - self.FINAL_EXPLORATION_PROB) *
                    (self.FINAL_EXPLORATION_FRAME - max(0, global_step - self.REPLAY_START_SIZE)) / self.FINAL_EXPLORATION_FRAME))


    def process_input(self, img):
        out = img[:195, :] # get only the playing area of the image
        r, g, b = out[:,:,0], out[:,:,1], out[:,:,2]
        out = r * (299./1000.) + r * (587./1000.) + b * (114./1000.)
        out = misc.imresize(out, (self.IMAGE_SIZE, self.IMAGE_SIZE), interp="bilinear")
        return out

    def update(self, step):
        loss = 0
        ep = 0

        if self.is_training == False:
            self._env.render()

        self.action, ep = self.get_next_action(step)

        # Execute action a_t in emulator and observe reward r_t and image x_t+1
        new_frame, reward, done, info = self._env.step(self.action)

        if self.lives > self._env.ale.lives():
           reward = -10.0
           self.lives = self._env.ale.lives()

        # Set s_t+1 = s_t, a_t, x_t+1 and preprocess s_t+1 = φ(st+1)
        new_observation = self.process_input(new_frame)
        new_observation = np.expand_dims(new_observation, axis=2)

        print "Old state shape:", self.state.shape

        # store transition (φ_t, a_t, r_t, φ_t+1) in D (experience decay collection)
        new_state = np.array(self.state[:, :, 1:], copy=True)
        new_state = np.append(new_state, new_observation, axis=2)
        print "New state shape:", new_state.shape

        self.check_previous_post_state(self.state, new_state)
        exit()


        new_state = np.append(self.state[:, :, 1:], new_observation, axis=2)
        experience_men = [self.state, self.action, reward, new_state, done]

        self.check_previous_post_state(self.state, new_state)

        #print experience_men[0].shape
        #print new_observation
        # plt.imshow(self.state)
        # plt.show()
        # exit()

        #self.display_experience_men(experience_men)
        exit()

        self.exp_replay_list.append(experience_men)

        # The agent has to select 4 actions between each SGD update
        if step % self.UPDATE_FREQUENCY == 0:
            # only train if done observing
            if step >= self.REPLAY_START_SIZE and self.is_training:

                if self.train_step == 0:
                    print "Starting training!"

                # Perform a gradient descent step on (y_j −Q(φ_j, a_j; θ))^2
                loss = self.train()

                if self.train_step % self.SAVE_MODEL_EVERY == 0:
                    save_path = self.saver.save(self._session, "./model/model.ckpt")
                    print("Model Saved in file:", save_path, "Last step:", step)

        self.state = new_state

        if done:
            # write log data to a csv file
            with open('log.csv', 'a+') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

                line = [str(step), str(self.game_number), str(self.train_step), \
                        str(loss), str(self.get_action_name(self.action)), str(ep), \
                        str(self.number_of_random_actions), \
                        str(self.number_of_systematic_actions)]
                writer.writerow(line)

            if self.game_number % 5 == 0:
                print self.rewards_distribution
                self.rewards_distribution = dict([(key, 0) for key in self.ACTION_NAMES])

            self.game_number += 1
            self.reset_debug_variables()
            _ = self.reset_env()

        return done, reward