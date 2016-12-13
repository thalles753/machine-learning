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
        print "--> Mode:", args.mode
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
        self.total_reward = 0

        self.reset_debug_variables()
        self.rewards_distribution = dict([(key, 0) for key in self.ACTION_NAMES])

        # start experience replay dataset
        self.exp_replay_list = deque(maxlen=self.args.replay_memory_size)

        # setup log file
        with open(self.args.mode + '_log.csv', 'w+') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            fieldnames = ['Global_step', 'Game_number', 'Traning_step', 'Loss', 'Last_action', 'Random_prob','#_of_Random_actions', '#_of_Conscious_actions', 'Total_reward']
            writer.writerow(fieldnames)

        # Tensorflow variables
        self._session = tf.Session()

        self.train_network = DQN.Network(args, self.NUM_ACTIONS, scope="train")

        if self.args.mode == "train":
            # target value function network
            self.target_network = DQN.Network(args, self.NUM_ACTIONS, scope="target")

        init = tf.initialize_all_variables()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        self.game_initial_setup()

        self._session.run(init)

        if self.args.mode == "test":
            self.restore_model()

    def copy_model_parameters(self, sess, estimator1, estimator2):
        """
        Copies the model parameters of one estimator to another.

        Args:
          sess: Tensorflow session instance
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        sess.run(update_ops)


    # normalize network input between 0 and 1
    def normalize_input(self, input):
        return np.divide(input, self.PIXEL_DEPTH)

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
        target_logits = self.target_network.predict(self._session, self.normalize_input(current_state))
        print target_logits

        train_logits = self.train_network.predict(self._session, self.normalize_input(current_state))
        print train_logits

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
                axarr[id][c].imshow(self.normalize_input(state[:,:,c]), cmap=plt.cm.Greys);
                axarr[id][c].set_title('Action: ' + self.get_action_name(memory[1]) + " Reward:" + str(memory[2]))
            id += 1
        plt.show()

    def train(self):
        # Sample random minibatch of transitions (s_i, a_i, r_i, s_i+1)
        # from the experience replay list
        mini_batch = random.sample(self.exp_replay_list, self.args.minibatch_size)
        # assert(len(mini_batch) == self.args.minibatch_size)

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
            self.copy_model_parameters(self._session, self.train_network, self.target_network)
            # self.target_network.copy_weights(self.train_network, self._session)
            print "Target network updated."
            print "(After) Are networks equal:", self.compare_train_target_net_random_weight()

        # this gives us the agents expected reward for each action we might take
        post_state_qvalues = self.target_network.predict(self._session, self.normalize_input(next_states))
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

        loss = self.train_network.update(self._session, self.normalize_input(previous_states),
                                         self.to_one_hot_vec(actions), agents_expected_reward, self.train_step)

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
            q_action = self.train_network.predict(self._session, self.normalize_input(current_state))
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


    # returns a 84 x 84 image tensor as described in the deep minds paper
    def process_input(self, img):
        out = img[:195, :] # get only the playing area of the image
        r, g, b = out[:,:,0], out[:,:,1], out[:,:,2]
        out = r * (299./1000.) + g * (587./1000.) + b * (114./1000.)
        out = misc.imresize(out, (self.args.screen_width, self.args.screen_height), interp="bilinear")
        return out

    # def process_rewards(self, reward):
    #     if self.lives > self._env.ale.lives():
    #         self.lives = self._env.ale.lives()
    #         return -10.0
    #     return reward

    def save_model(self):
        if self.args.mode == "train":
            save_path = self.saver.save(self._session, "./model/model.ckpt")
            print("Model saved in file: %s" % save_path)

    def update(self, step):
        loss = 0
        ep = 0

        if self.args.render:
            self._env.render()

        action, ep = self.get_next_action(step)

        # Execute action a_t in emulator and observe reward r_t and image x_t+1
        new_frame, reward, done, info = self._env.step(action)
        self.total_reward += reward

        if done:
            reward = -10.0

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
                        print "Start training!"

                    # Perform a gradient descent step on (y_j −Q(φ_j, a_j; θ))^2
                    loss = self.train()

        if done:
            # write log data to a csv file
            with open(self.args.mode + '_log.csv', 'a+') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

                line = [str(step), str(self.game_number), str(self.train_step), \
                        str(loss), str(self.get_action_name(action)), str(ep), \
                        str(self.number_of_random_actions), \
                        str(self.number_of_systematic_actions), str(self.total_reward)]
                writer.writerow(line)

            self.game_number += 1

            if self.game_number % self.args.average_reward_stats_per_game == 0:
                average_reward = self.total_reward / self.args.average_reward_stats_per_game
                self.train_network.update_average_reward(self._session, average_reward, self.game_number)
                self.total_reward = 0

                print self.rewards_distribution
                self.rewards_distribution = dict([(key, 0) for key in self.ACTION_NAMES])

            self.reset_debug_variables()
            self.reset_env()

        self.state = next_state
        return done
