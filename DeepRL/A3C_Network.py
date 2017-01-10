# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class A3C_Network:
    def __init__(self, args, output_size, scope):
        self.args = args
        self.scope = scope
        input_shape = (args.screen_width, args.screen_height, args.agent_history_length) # 84 x 84 x 4
        with tf.variable_scope(scope):
            self.output_size = output_size
            self._input = tf.placeholder(tf.float32,
                                         shape=(None,) + (input_shape[0], input_shape[1], input_shape[2]),
                                         name="input_state")
            self._target = tf.placeholder(tf.float32, [None], name="input_targets")
            self._action = tf.placeholder(tf.float32, [None, output_size], name="input_actions_one_hot")
            self._build_graph()

            summary_path = './summary/' + self.args.game_name + "/" + self.args.mode + "/"
            self.train_writer = tf.train.SummaryWriter(summary_path)
            self.total_reward_ph = tf.placeholder(tf.float32, [], name="total_reward_placeholder")
            self.total_reward_summary = tf.scalar_summary(self.scope + '_total_reward', self.total_reward_ph)

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

        self.value_func_prediction = slim.fully_connected(fc1, self.output_size, activation_fn=None, biases_initializer=None)

        # softmax output with one entry per action representing the probability of taking an action
        self.policy_predictions = slim.fully_connected(fc1, self.output_size, activation_fn=tf.nn.softmax, biases_initializer=None)


        # Only the worker network need ops for loss functions and gradient updating.
        # if self.scope != 'global':
        #
        #     self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
        #     self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
        #
        #     self.responsible_outputs = tf.reduce_sum(self.policy_predictions * self.actions, [1])
        #
        #     #Loss functions
        #     self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
        #     self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        #     self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
        #     self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
        #
        #     #Get gradients from local network using local losses
        #     local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        #     self.gradients = tf.gradients(self.loss,local_vars)
        #     self.var_norms = tf.global_norm(local_vars)
        #     grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
        #
        #     #Apply local gradients to global network
        #     global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        #     self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))



        # Get the value function predictions for the chosen actions only
        readout_action = tf.reduce_sum(tf.mul(self.value_func_prediction, self._action), reduction_indices=1)

        diff = self._target - readout_action
        diff_clipped = tf.clip_by_value(diff, -self.args.clip_error, self.args.clip_error)
        self.loss = tf.reduce_mean(tf.square(diff_clipped))

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, epsilon=0.001)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
        self.train_operation = optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        self.merged = tf.merge_summary([
            tf.scalar_summary('loss', self.loss),
            tf.histogram_summary("q_values_history", self.value_func_prediction),
            tf.scalar_summary('max_q_value', tf.reduce_max(self.value_func_prediction))
        ])

        self.action_ids = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.predictions_by_id = tf.gather_nd(self.value_func_prediction, self.action_ids)

    def predict(self, sess, states):
        return sess.run(self.value_func_prediction, {self._input: states})

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