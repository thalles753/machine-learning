# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from constants import GLOBAL_NETWORK_NAME

# Asynchronous Advantage Actor-Critic (A3C)
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

            if scope != GLOBAL_NETWORK_NAME:
                # create sync and losses operations
                self.sync = self._prepare_sync_ops()
                self._prepare_loss_ops()

            summary_path = './summary/' + self.args.game_name + "/" + self.args.mode + "/"
            self.train_writer = tf.train.SummaryWriter(summary_path)

    def update_epsode_average_reward(self, sess, episode_reward, game_number):
        summary = tf.Summary(value=episode_reward)
        self.train_writer.add_summary(summary, game_number)

    # Build network as described in (Mnih et al., 2013)
    def _build_graph(self):

        normalized_input = tf.div(self._input, 255.0)

        conv1 = slim.conv2d(normalized_input, 16, [8, 8], activation_fn=tf.nn.relu,
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

    def _prepare_loss_ops(self):
        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None], name="r-v_values")

        policy = self.policy_predictions

        # R (input for value)
        self.r = tf.placeholder("float", [None], name="R_values")

        # avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(policy) #log π(a_i|s_i; θ?)

        # policy entropy
        entropy = - tf.reduce_sum(policy * log_pi)

        # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
        policy_loss = - tf.reduce_sum(tf.reduce_sum(log_pi * self._action, reduction_indices=1) * self.td)

        # value loss function (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        value_loss = 0.5 * tf.reduce_sum(tf.square(self.value_func_prediction - self.r))

        bs = tf.to_float(tf.shape(self._input)[0])

        # gradients of policy and value are summed up
        self.total_loss = 0.5 * value_loss + policy_loss - entropy * self.args.entropy_regularization

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)
        self.gradients = tf.gradients(self.total_loss, local_vars)

        # get variables from the global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NETWORK_NAME)
        grads, _ = tf.clip_by_global_norm(self.gradients, 40.0)

        # apply the gradients
        self.apply_gradients = self.trainer.apply_gradients(zip(grads, global_vars))

        self.merged = tf.merge_summary([
            tf.scalar_summary('loss', self.total_loss / bs),
            tf.scalar_summary("value_loss", value_loss / bs),
            tf.scalar_summary('policy_loss', policy_loss / bs)
        ])

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)

    def update_gradients(self, sess, batch_states, batch_actions_one_hot, batch_td, batch_R, train_step, thread_id):
        summaries, grads = sess.run([self.merged, self.apply_gradients],
        feed_dict = {
            self._input: batch_states,
            self._action: batch_actions_one_hot,
            self.td: batch_td,
            self.r: batch_R,
        })

        if train_step % self.args.update_tf_board == 0 and thread_id == 0:
            self.train_writer.add_summary(summaries, train_step)

    def _prepare_sync_ops(self):
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NETWORK_NAME)
        local_vars = self.get_variables()

        sync_ops = []

        for(src_var, dst_var) in zip(global_vars, local_vars):
            sync_op = tf.assign(dst_var, src_var)
            sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def sync_local_net(self, sess):
        sess.run(self.sync)
        # print "Local network:", self.scope_name, "successfully updaded!"