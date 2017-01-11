# -*- coding: utf-8 -*-

import gym
from A3C_Network import A3C_Network
from Utils import copy_model_parameters
from Utils import process_input
import numpy as np
import tensorflow as tf

class Worker:
    # each worker has its own network and its own environment
    def __init__(self, args, thread_id, model_path, global_episodes, global_network, trainer):
        print "Creating worker: ", thread_id
        self.args = args
        self.thread_id = thread_id
        self.trainer = trainer
        self.model_path = model_path
        self.global_episodes = global_episodes
        self.global_network = global_network
        self.scope = "worker_" + str(thread_id)
        self.state = None

        # creates own worker agent environment
        self._env = gym.make(self.args.game_name)

        # get the number of available actions
        self.n_actions = self._env.action_space.n

        # reset openai gym environment and create first state
        self.reset_env()
        self.game_initial_setup()

        # creates own worker agent network
        self.network = A3C_Network(args=args, output_size=self.n_actions, trainer=trainer, scope=self.scope)

        # with tf.device("/cpu:0"):
        #     self.gradients = tf.gradients(self.network.total_loss, self.network.get_vars())
        #     # grads_and_vars = self.trainer.compute_gradients(self.network.total_loss, self.network.get_vars())
        #
        # # get variables from the global network
        # global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        # grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
        #
        # # apply the gradients
        # self.apply_gradients = trainer.apply_gradients(zip(grads, global_vars))

    # for the first step, the state is the same frame screen repeated [agent_history_length] times
    def game_initial_setup(self):
        frame, _, _, _ = self._env.step(0)
        processed_frame = process_input(frame, self.args.screen_width, self.args.screen_height)
        self.state = np.stack(tuple(processed_frame for _ in range(self.args.agent_history_length)), axis=2)
        assert(self.state.shape == (self.args.screen_width, self.args.screen_height, self.args.agent_history_length))
        self._env.frameskip = self.args.frame_skip
        print "Game setup and ready to go!"

    def reset_env(self):
        _ = self._env.reset()
        self.lives = self._env.ale.lives()

    def choose_action_randomly(self, action_distributions):
        a = np.random.choice(action_distributions, p=action_distributions)
        action_id = np.argmax(action_distributions == a)
        return action_id

    def work(self, sess, coordinator):
        print "Thread:", self.thread_id, "has started."

        episode_count = sess.run(self.global_episodes)

        t = 0

        while (True):
            print "Time step:", t
            t_start = t;

            # reset the worker's local network weights to be the same of the global network
            copy_model_parameters(sess, self.global_network, self.network)

            action_counter = 0
            experiences = []

            while True:

                # Perform action at according to policy π(at|st; θ')
                action = self.network.predict_policy(sess, np.expand_dims(self.state, axis=0))
                action_index = self.choose_action_randomly(action)

                observation, reward, is_terminal, info = self._env.step(action_index)

                # create new state using
                new_observation = np.expand_dims(process_input(observation), axis=2) # 84 x 84 x 1
                next_state = np.array(self.state[:, :, 1:], copy=True)
                next_state = np.append(next_state, new_observation, axis=2)

                # store experiences
                ex = [self.state, action_index, reward, next_state, is_terminal]
                experiences.append(ex)

                # update local thread counter
                t += 1

                self.state = next_state

                action_counter += 1

                # TODO: update global counter

                if t - t_start == self.args.tmax:
                    break

            R = 0.0
            if not is_terminal:
                R = self.network.predict_values(sess, np.expand_dims(self.state, axis=0))

            self.compute_and_accumulate_rewards(R, experiences, sess)

    def compute_and_accumulate_rewards(self, R, experiences, sess):
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

            self.network.update_gradients(sess, batch_states, batch_actions_one_hot, batch_td, batch_R)

            # sess.run(self.apply_gradients,
            # feed_dict = {
            #     self.network._input: batch_states,
            #     self.network._action: batch_actions_one_hot,
            #     self.network.td: batch_td,
            #     self.network.r: batch_R,
            # })