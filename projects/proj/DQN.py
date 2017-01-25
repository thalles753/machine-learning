# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class Network:
    def __init__(self, args, output_size, scope):
        self.args = args
        self.scope = scope
        input_shape = (args.screen_width, args.screen_height, args.agent_history_length) # 84 x 84 x 4
        with tf.variable_scope(scope):
            self.output_size = output_size
            self._input = tf.placeholder(tf.float32,
                                         shape=(None,) + (input_shape[0], input_shape[1], input_shape[2]),
                                         name="input_images")
            self._target = tf.placeholder(tf.float32, [None], name="input_targets")
            self._action = tf.placeholder(tf.float32, [None, output_size], name="input_actions")
            self._build_graph()

            summary_path = './summary/' + self.args.game_name + "/" + self.args.mode + "/"
            self.train_writer = tf.train.SummaryWriter(summary_path)
            self.total_reward_ph = tf.placeholder(tf.float32, [], name="total_reward_placeholder")
            self.total_reward_summary = tf.scalar_summary(self.scope + '_total_reward', self.total_reward_ph)

    def update_average_reward(self, sess, average_reward, game_number):
        reward_summary = sess.run(self.total_reward_summary,
                                  feed_dict={self.total_reward_ph: average_reward})
        self.train_writer.add_summary(reward_summary, game_number)

    def _build_graph(self):

        conv1 = slim.conv2d(self._input, 32, [8, 8], activation_fn=tf.nn.relu,
                            padding='VALID', stride=4, biases_initializer=None)

        conv2 = slim.conv2d(conv1, 64, [4, 4], activation_fn=tf.nn.relu,
                            padding='VALID', stride=2, biases_initializer=None)

        conv3 = slim.conv2d(conv2, 64, [3, 3], activation_fn=tf.nn.relu,
                            padding='VALID', stride=1, biases_initializer=None)

        flattened = slim.flatten(conv3)

        fc1 = slim.fully_connected(flattened, 512, activation_fn=tf.nn.relu, biases_initializer=None)

        self.predictions = slim.fully_connected(fc1, self.output_size, activation_fn=None, biases_initializer=None)

        # Get the predictions for the chosen actions only
        readout_action = tf.reduce_sum(tf.mul(self.predictions, self._action), reduction_indices=1)

        diff = self._target - readout_action
        diff_clipped = tf.clip_by_value(diff, -self.args.clip_error, self.args.clip_error)
        self.loss = tf.reduce_mean(tf.square(diff_clipped))

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, epsilon=0.01)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
        self.train_operation = optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        self.merged = tf.merge_summary([
            tf.scalar_summary('loss', self.loss),
            tf.histogram_summary("q_values_history", self.predictions),
            tf.scalar_summary('max_q_value', tf.reduce_max(self.predictions))
        ])

        self.action_ids = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.predictions_by_id = tf.gather_nd(self.predictions, self.action_ids)

    def predict(self, sess, states):
        return sess.run(self.predictions, {self._input: states})

    def update(self, sess, states, actions_one_hot, targets, train_step):
        # Perform a gradient descent step on (y_j − Q(s_j, a_j; θ))^2
        summary, _, loss = sess.run([self.merged, self.train_operation, self.loss], feed_dict={
            self._input: states,
            self._action: actions_one_hot,
            self._target: targets})

        if train_step % self.args.update_tf_board == 0:
            print "Summary data has been written!!"
            self.train_writer.add_summary(summary, train_step)
        return loss

    def get_predictions_by_action_ids(self, sess, states, actions_ids):
        return self.predictions_by_id.eval(
            {self._input: states, self.action_ids: actions_ids}, session=sess)