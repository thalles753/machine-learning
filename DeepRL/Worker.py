# -*- coding: utf-8 -*-

import gym
from A3C_Network import A3C_Network
from Utils import process_input
import numpy as np
import os
import tensorflow as tf
from Utils import display_transition

class Worker:
    # each worker has its own network and its own environment
    def __init__(self, args, thread_id, model_path, global_episodes, global_network, trainer, lock):
        print "Creating worker: ", thread_id
        self.args = args
        self.thread_id = thread_id
        self.trainer = trainer
        self.model_path = model_path
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.global_network = global_network
        self.scope = "worker_" + str(thread_id)
        self.state = None
        self.lives = None
        self.lock = lock

        # with self.lock:
        # creates own worker agent environment
        self._env = gym.make(self.args.game_name)

        # creates own worker agent environment
        self._env = gym.make(self.args.game_name)

        # get the number of available actions
        self.n_actions = self._env.action_space.n

        # reset openai gym environment and create first state
        self.reset_game_env()
        self.game_initial_setup()

        # creates own worker agent network
        self.network = A3C_Network(args=args, output_size=self.n_actions, trainer=trainer, scope=self.scope)

    # for the first step, the state is the same frame screen repeated [agent_history_length] times
    def game_initial_setup(self):
        frame, _, _, _ = self._env.step(0)
        processed_frame = process_input(frame, self.args.screen_width, self.args.screen_height)
        self.state = np.stack(tuple(processed_frame for _ in range(self.args.agent_history_length)), axis=2)
        assert(self.state.shape == (self.args.screen_width, self.args.screen_height, self.args.agent_history_length))
        self._env.frameskip = self.args.frame_skip
        print "Game setup and ready to go!"

    def choose_action_randomly(self, action_distributions):
        a = np.random.choice(action_distributions, p=action_distributions)
        action_id = np.argmax(action_distributions == a)
        return action_id

    def reset_game_env(self):
        # with self.lock:
        self._env.reset()
        self.lives = self._env.ale.lives()

    def process_lives(self):
        terminal = False
        if self._env.ale.lives() > self.lives:
            self.lives = self._env.ale.lives()

        # Loosing a life will trigger a terminal signal in training mode.
        # We assume that a "life" IS an episode during training, but not during testing
        elif self._env.ale.lives() < self.lives:
            self.lives = self._env.ale.lives()
            terminal = True
        return terminal

    def work(self, sess, coordinator):
        print "Thread:", self.thread_id, "has started."

        episode_count = sess.run(self.global_episodes)

        episode_number = 0
        total_reward = 0
        total_reward_list = []
        t = 0

        for train_step in range(25000000):
            # print "Time step:", t
            t_start = t;

            # print "--", self.network.scope_name, "-- Are nets equal:", self.compare_global_and_local_networks(sess)
            # reset the worker's local network weights to be the same of the global network
            self.network.sync_local_net(sess)
            # print "--", self.network.scope_name, "-- Are nets equal:", self.compare_global_and_local_networks(sess)

            experiences = []

            while True:

                # Perform action at according to policy π(a_t|s_t; θ')
                action = self.network.predict_policy(sess, np.expand_dims(self.state, axis=0))
                action_index = self.choose_action_randomly(action)

                # with self.lock:
                # self._env.render()
                observation, reward, is_terminal, info = self._env.step(action_index)

                total_reward += reward

                if self.args.mode == "train":
                    is_terminal = self.process_lives()

                if is_terminal:
                    reward = -10.0

                # create new state using
                new_observation = np.expand_dims(process_input(observation), axis=2) # 84 x 84 x 1
                next_state = np.array(self.state[:, :, 1:], copy=True)
                next_state = np.append(next_state, new_observation, axis=2)

                # clip the rewards
                clipped_reward = np.clip(reward, self.args.min_reward, self.args.max_reward)

                # store experiences
                ex = [self.state, action_index, clipped_reward, next_state, is_terminal]
                # display_transition(self._env.get_action_meanings(), ex)

                experiences.append(ex)

                # update local thread counter
                t += 1

                self.state = next_state

                # TODO: update global counter

                if t - t_start == self.args.tmax or is_terminal:
                    break

            R = 0.0
            if not is_terminal:
                R = self.network.predict_values(sess, np.expand_dims(self.state, axis=0))

            self.compute_and_accumulate_rewards(R, experiences, sess, train_step)

            if is_terminal:

                if self.thread_id == 0:
                    total_reward_list.append(tf.Summary.Value(tag="episode_average_reward", simple_value=total_reward))

                if self.thread_id == 0:
                    self.network.update_epsode_average_reward(sess, total_reward_list, episode_number)
                    print "Summary data has been written."
                    total_reward_list = []

                total_reward = 0
                episode_number += 1

                if self.thread_id == 0:
                    print "Epsode #", episode_number, "has finished."

                # reset environment
                self.reset_game_env()

        if self.thread_id == 0 and self.args.mode == "train":
            self.save_model()

    def compute_and_accumulate_rewards(self, R, experiences, sess, train_step):
        previous_states = [d[0] for d in experiences]
        actions = [d[1] for d in experiences]
        rewards = [d[2] for d in experiences]
        next_states = [d[3] for d in experiences]
        terminals = [d[4] for d in experiences]

        previous_states.reverse()
        actions.reverse()
        rewards.reverse()
        next_states.reverse()
        terminals.reverse()

        batch_states = []
        batch_actions_one_hot = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(state, action, reward) in zip(previous_states, actions, rewards):
            R = reward + self.args.discount_factor * R
            td = R - self.network.predict_values(sess, np.expand_dims(state, axis=0)) # (R - V(si; θ'v)
            a = np.zeros([self.n_actions])
            a[action] = 1

            batch_states.append(state)
            batch_actions_one_hot.append(a)
            batch_td.append(td)
            batch_R.append(R)

        self.network.update_gradients(sess, batch_states, batch_actions_one_hot, batch_td, batch_R, train_step, self.thread_id)

    def compare_global_and_local_networks(self, sess):
        current_state = np.expand_dims(self.state, axis=0)

        global_net_policy = self.global_network.predict_policy(sess, current_state)
        local_nets_policy = self.network.predict_policy(sess, current_state)

        if np.array_equal(global_net_policy, local_nets_policy):
            return True
        else:
            return False

    def save_model(self):
        model_path = "./models/" + self.args.game_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print "Model folder created."
        save_path = self.saver.save(self._session, model_path + "/" + "model.ckpt")
        print("Model saved in file: %s" % save_path)