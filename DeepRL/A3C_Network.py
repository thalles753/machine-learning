# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class A3C_Network:
    def __init__(self, args, output_size, trainer, scope):
        self.args = args
        self.trainer = trainer
        self.scope_name = scope
        input_shape = (args.screen_width, args.screen_height, args.agent_history_length) # 84 x 84 x 4
        with tf.variable_scope(scope):
            self.output_size = output_size
            self._input = tf.placeholder(tf.float32,
                                         shape=(None,) + (input_shape[0], input_shape[1], input_shape[2]),
                                         name="input_state")
            self._target = tf.placeholder(tf.float32, [None], name="input_targets")
            self._action = tf.placeholder(tf.float32, [None, output_size], name="input_actions_one_hot")
            self._build_graph()

            if scope != "global":
                self.prepare_loss_ops()

            summary_path = './summary/' + self.args.game_name + "/" + self.args.mode + "/"
            self.train_writer = tf.train.SummaryWriter(summary_path)
            self.total_reward_ph = tf.placeholder(tf.float32, [], name="total_reward_placeholder")
            self.total_reward_summary = tf.scalar_summary(self.scope_name + '_total_reward', self.total_reward_ph)

    def update_average_reward(self, sess, average_reward, game_number):
        reward_summary = sess.run(self.total_reward_summary,
                                  feed_dict={self.total_reward_ph: average_reward})
        self.train_writer.add_summary(reward_summary, game_number)

    # Build network as described in (Mnih et al., 2013)
    def _build_graph(self):

        conv1 = slim.conv2d(self._input, 16, [8, 8], activation_fn=tf.nn.relu,
                            padding='VALID', stride=4, biases_initializer=None)

        conv2 = slim.conv2d(conv1, 32, [4, 4], activation_fn=tf.nn.relu,
                            padding='VALID', stride=2, biases_initializer=None)

        flattened = slim.flatten(conv2)

        fc1 = slim.fully_connected(flattened, 256, activation_fn=tf.nn.relu, biases_initializer=None)

        self.value_func_prediction = slim.fully_connected(fc1, 1, activation_fn=None, biases_initializer=None)

        # softmax output with one entry per action representing the probability of taking an action
        self.policy_predictions = slim.fully_connected(fc1, self.output_size, activation_fn=tf.nn.softmax, biases_initializer=None)

    def predict_values(self, sess, states):
        return sess.run(self.value_func_prediction, {self._input: states})[0][0]

    def predict_policy(self, sess, states):
        return sess.run(self.policy_predictions, {self._input: states})[0]

    def prepare_loss_ops(self):
        with tf.device("/cpu:0"):

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None], name="Td_values")

            # R (input for value)
            self.r = tf.placeholder("float", [None], name="R_values")

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.policy_predictions, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.policy_predictions * log_pi, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( log_pi, self._action ), reduction_indices=1 ) * self.td + entropy * self.args.entropy_regularization)

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.value_func_prediction)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss

            self.gradients = tf.gradients(self.total_loss, self.get_vars())
            # grads_and_vars = self.trainer.compute_gradients(self.network.total_loss, self.network.get_vars())

            # get variables from the global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # apply the gradients
            self.apply_gradients = self.trainer.apply_gradients(zip(grads, global_vars))

    def get_vars(self):
        print "Getting variables from:", self.scope_name, "Network"
        model_variables = slim.get_model_variables(self.scope_name)
        return model_variables

    def update_gradients(self, sess, batch_states, batch_actions_one_hot, batch_td, batch_R):
        sess.run(self.apply_gradients,
        feed_dict = {
            self._input: batch_states,
            self._action: batch_actions_one_hot,
            self.td: batch_td,
            self.r: batch_R,
        })