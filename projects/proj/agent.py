# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from collections import deque
from scipy import misc
import DQN
import random
import csv
import matplotlib.pyplot as plt


class LearningAgent:
    def __init__(self, env, args, mode="train"):
        # game variables and constants
        self.args = args
        self._env = env
        _ = self._env.reset()
        self.lives = self._env.ale.lives()
        self.ACTION_NAMES = self._env.get_action_meanings()
        self.NUM_ACTIONS = env.action_space.n
        self.state = None
        self.train_step = 0
        self.PIXEL_DEPTH = 255.0
        self.game_number = 0

        self.reset_debug_variables()
        self.rewards_distribution = dict([(key, 0) for key in self.ACTION_NAMES])

        # start experience replay dataset
        self.exp_replay_list = deque(maxlen=self.args.replay_memory_size)

        self.SAVE_MODEL_EVERY = 10000
        self.UPDATE_TF_BOARD_EVERY = 1000
        self.SHOW_DEBUG = 5

        # setup log file
        with open(self.args.mode + 'log.csv', 'w+') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            fieldnames = ['Global_step', 'Game_number', 'Traning_step', 'Loss', 'Last_action', 'Random_prob','#_of_Random_actions', '#_of_Conscious_actions']
            writer.writerow(fieldnames)

        # Tensorflow variables
        self._session = tf.Session()
        network_input_shape = (self.args.screen_width, self.args.screen_height, self.args.agent_history_length) # 84 x 84 x 4

        # action-value function Q
        with tf.variable_scope("train") as self.target_scope:
            self.train_network = DQN.Network(network_input_shape, self.NUM_ACTIONS)

        # target action-value function Q_hat
        with tf.variable_scope("target") as self.train_scope:
            self.target_network = DQN.Network(network_input_shape, self.NUM_ACTIONS)

        if self.args.mode == "train":
            self._target = tf.placeholder(tf.float32, [None], name="input_targets")
            self._action = tf.placeholder(tf.float32, [None, self.NUM_ACTIONS], name="input_actions")
            readout_action = tf.reduce_sum(tf.mul(self.train_network.logits, self._action), reduction_indices=1)

            diff = self._target - readout_action
            diff_clipped = tf.clip_by_value(diff, -self.args.clip_error, self.args.clip_error)
            self.loss = tf.reduce_mean(tf.square(diff_clipped))
            loss_summary = tf.scalar_summary('loss', self.loss)

            optimizer = tf.train.RMSPropOptimizer(learning_rate = self.args.learning_rate, epsilon=0.01, momentum=0.95, decay=0.95)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
            self.train_operation = optimizer.minimize(self.loss)

            self.merged = tf.merge_summary([loss_summary])
            self.train_writer = tf.train.SummaryWriter('./summary', self._session.graph)

        self._session.run(tf.initialize_all_variables())

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        self.game_initial_setup()

        if self.args.mode == "test":
            self.restore_model()

    def restore_model(self):
        self.saver.restore(self._session, "./model/model.ckpt")
        print("Model restored.")

    def reset_debug_variables(self):
        self.number_of_random_actions = 0
        self.number_of_systematic_actions = 0

    def to_one_hot_vec(self, actions):
        one_hot = np.zeros((self.args.minibatch_size, self.NUM_ACTIONS))
        one_hot[np.arange(self.args.minibatch_size), actions] = 1
        return one_hot

    def compare_train_target_net_random_weight(self):
        current_state = np.expand_dims(self.state, axis=0)
        target_logits = self._session.run(self.target_network.logits, feed_dict={self.target_network._input: current_state})
        train_logits = self._session.run(self.train_network.logits, feed_dict={self.train_network._input: current_state})
        if np.array_equal(target_logits,train_logits):
            return True
        else:
            return False

    # debug routine
    def check_prev_post_state(self, prev, post):
        a = np.array_equal(prev[:, :, 1], post[:, :, 0])
        b = np.array_equal(prev[:, :, 2], post[:, :, 1])
        c = np.array_equal(prev[:, :, 3], post[:, :, 2])

        if (not a or not b or not c):
            raise "The post state does not comply with the previous state."
        return True

    # debug routine
    def display_transition(self, memory):
        # displya post states
        states = [memory[0], memory[3]]
        id = 0
        f, axarr = plt.subplots(2, 4, figsize=(18,8))
        for state in states:
            for c in range(0,4):
                axarr[id][c].imshow(state[:,:,c], cmap=plt.cm.Greys);
                axarr[id][c].set_title('Action: ' + self.get_action_name(memory[1]) + " Reward:" + str(memory[2]))
            id += 1
        plt.show()

    def train(self):
        # Sample random minibatch of transitions (s_i, a_i, r_i, s_i+1)
        # from the experience replay list
        mini_batch = random.sample(self.exp_replay_list, self.args.minibatch_size)
        assert(len(mini_batch) == self.args.minibatch_size)

        previous_states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        next_states = [d[3] for d in mini_batch]
        terminals = [d[4] for d in mini_batch]

        # clip rewards between -1 and 1
        rewards = np.clip(rewards, a_min=-1.0, a_max=1.0)

        if self.train_step % self.args.target_network_update_frequency == 0:
            print "(Before) Are networks equal:", self.compare_train_target_net_random_weight()
            # update target network
            self.target_network.copy_weights(self.train_network, self._session)
            print "Target network updated."
            print "(After) Are networks equal:", self.compare_train_target_net_random_weight()

        # this gives us the agents expected reward for each action we might take
        post_state_qvalues = self._session.run(self.target_network.logits, feed_dict={self.target_network._input: next_states})
        max_post_state_qvalues = np.max(post_state_qvalues, 1)

        agents_expected_reward = []
        for i in range(self.args.minibatch_size):
            if terminals[i]:
                # this was a terminal state so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                # compute r_j + γ max Q^(s_j+1, a; θ-),
                #    where:
                #      s_j is the next state
                #      θ- are the CNNs learned weights
                agents_expected_reward.append(
                    rewards[i] + self.args.discount_factor * max_post_state_qvalues[i])

        # Perform a gradient descent step on (y_j − Q(s_j, a_j; θ))^2
        summary, _, loss = self._session.run([self.merged, self.train_operation, self.loss], feed_dict={
            self.train_network._input: previous_states,
            self._action: self.to_one_hot_vec(actions),
            self._target: agents_expected_reward})


        if self.train_step % self.UPDATE_TF_BOARD_EVERY == 0:
            print "Summary data has been written!!"
            self.train_writer.add_summary(summary, self.train_step)

        self.train_step += 1
        return loss

    def reset_env(self):
        _ = self._env.reset()
        self.lives = self._env.ale.lives()

    def get_action_name(self, action):
        return self.ACTION_NAMES[action]

    # for the first step, the state is the same frame screen repeated [agent_history_length] times
    def game_initial_setup(self):
        frame, _, _, _ = self._env.step(0)
        processed_frame = self.process_input(frame)
        self.state = np.stack(tuple(processed_frame for _ in range(self.args.agent_history_length)), axis=2)
        assert(self.state.shape == (self.args.screen_width, self.args.screen_height, self.args.agent_history_length))

        for i in range(0,4):
            for j in range(0,4):
                if np.array_equal(self.state[:,:,i], self.state[:,:,j]) == False:
                    exit("Not equal")

        self._env.frameskip = self.args.frame_skip
        print "Game setup and ready to go!"

    def get_next_action(self, step):
        # With probability p select a random action (a) otherwise select a = max Q(φ(st), a; θ)
        p = self.get_exploration_probability(step)

        if random.random() < p:
            out = self._env.action_space.sample()
            # print("Random action:", self.get_action_name(out))
            self.number_of_random_actions += 1
        else:
            current_state = np.expand_dims(self.state, axis=0)
            q_action = self._session.run(self.train_network.logits, feed_dict={self.train_network._input: current_state})
            out = np.argmax(q_action)
            # print("Predicted action:", self.get_action_name(out))
            self.number_of_systematic_actions += 1

        self.rewards_distribution[self.get_action_name(out)] += 1
        return out, p

    # perform a linear decay operation
    def get_exploration_probability(self, global_step):
        return (self.args.final_exploration +
                max(0, (self.args.initial_exploration - self.args.final_exploration) *
                    (self.args.final_exploration_frame - max(0, global_step - self.args.replay_start_size)) / self.args.final_exploration_frame))


    # returns a 84 x 84 x 4 image tensor as described in the deep minds paper
    def process_input(self, img):
        out = img[:195, :] # get only the playing area of the image
        r, g, b = out[:,:,0], out[:,:,1], out[:,:,2]
        out = r * (299./1000.) + r * (587./1000.) + b * (114./1000.)
        out = misc.imresize(out, (self.args.screen_width, self.args.screen_height), interp="bilinear")
        return out

    def process_rewards(self, reward):
        if self.lives > self._env.ale.lives():
            self.lives = self._env.ale.lives()
            return -10.0
        return reward

    def update(self, step):
        loss = 0
        ep = 0

        action, ep = self.get_next_action(step)

        # Execute action a_t in emulator and observe reward r_t and image x_t+1
        new_frame, reward, done, info = self._env.step(action)

        reward = self.process_rewards(reward)

        new_observation = np.expand_dims(self.process_input(new_frame), axis=2) # 84 x 84 x 1
        next_state = np.array(self.state[:, :, 1:], copy=True)
        next_state = np.append(next_state, new_observation, axis=2)

        if self.args.mode == "train":
            experience = [self.state, action, reward, next_state, done]

            # print self.check_prev_post_state(self.state, new_state)
            # self.display_transition(experience)

            # store transition (φ_t, a_t, r_t, φ_t+1) in D (experience decay collection)
            self.exp_replay_list.append(experience)

            # The agent has to select 4 actions between each SGD update
            if step % self.args.update_frequency == 0:
                # only train if done observing
                if step >= self.args.replay_start_size:

                    if self.train_step == 0:
                        print "Starting training!"

                    # Perform a gradient descent step on (y_j −Q(φ_j, a_j; θ))^2
                    loss = self.train()

                    if self.train_step % self.SAVE_MODEL_EVERY == 0:
                        save_path = self.saver.save(self._session, "./model/model.ckpt")
                        print("Model Saved in file:", save_path, "Last step:", step)

        if done:
            # write log data to a csv file
            with open(self.args.mode + '_log.csv', 'a+') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

                line = [str(step), str(self.game_number), str(self.train_step), \
                        str(loss), str(self.get_action_name(action)), str(ep), \
                        str(self.number_of_random_actions), \
                        str(self.number_of_systematic_actions)]
                writer.writerow(line)

            if self.game_number % self.SHOW_DEBUG == 0:
                print self.rewards_distribution
                self.rewards_distribution = dict([(key, 0) for key in self.ACTION_NAMES])

            self.game_number += 1
            self.reset_debug_variables()
            self.reset_env()

        self.state = next_state
        return done, reward