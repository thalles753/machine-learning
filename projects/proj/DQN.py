# -*- coding: utf-8 -*-

import tensorflow as tf

class Network:
    def __init__(self, input_shape, output_size):
        self.SEED = 999
        self.input_shape = input_shape
        self.outpt_size = output_size
        self.weights = {}
        self._input = tf.placeholder(tf.float32, shape=(None,) + (self.input_shape[0],self.input_shape[1],self.input_shape[2]), name="input_images")
        self.create_model_variables()
        self.logits = self.model()
        self.create_weight_cp_ops()


    def get_training_var_data(self, sess):
        for (var_name, var) in self.weights.iteritems():
            if var_name == 'train/conv1:0' or var_name == 'target/conv1:0':
                return var.eval(session=sess)


    def weight_variable(self, name, shape):
      return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))

    def weight_conv_variable(self, name, shape):
      return tf.get_variable(name=name, shape=shape,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.SEED))

    def bias_variable(self, name, shape):
      initial = tf.zeros_initializer(shape=shape, dtype=tf.float32)
      return tf.Variable(initial, name=name)

    def conv2d(self, x, W, strides=[1, 1, 1, 1], padding='SAME'):
      return tf.nn.conv2d(x, W, strides=strides, padding=padding)

    def max_pool(self, x, strides=[1, 2, 2, 1]):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=strides, padding='SAME')

    def copy_weights(self, other_net, sess):
        other_var_dir = other_net.weights
        for (other_var_name, other_var) in other_var_dir.iteritems():
            var_name = "target" + '/' + other_var_name.split('/', 1)[1]
            other_var_eval = other_var.eval(session=sess)
            sess.run(self.weight_copy_ops[var_name], feed_dict={self.weight_placeholders[var_name]: other_var_eval})

    def create_weight_cp_ops(self):
        self.weight_placeholders = {}
        for var_name in self.weights:
            self.weight_placeholders[var_name] = tf.placeholder(tf.float32)
        self.weight_copy_ops = {}
        for (var_name, var_placeholder) in self.weight_placeholders.iteritems():
            self.weight_copy_ops[var_name] = self.weights[var_name].assign(var_placeholder)

    def create_model_variables(self):
        # The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the
        # input image and applies a rectifier nonlinearity
        CONV1_DEPTH = 32
        self.W_conv1 = self.weight_conv_variable("conv1", [8, 8, self.input_shape[2], CONV1_DEPTH])
        self.b_conv1 = self.bias_variable('conv1_bias', [CONV1_DEPTH])
        self.weights[self.W_conv1.name] = self.W_conv1
        self.weights[self.b_conv1.name] = self.b_conv1

        # The second hidden layer convolves 64 filters of 4 x 4
        # with stride 2, again followed by a rectifier nonlinearity
        CONV2_DEPTH = 64
        self.W_conv2 = self.weight_conv_variable("conv2", [4, 4, CONV1_DEPTH, CONV2_DEPTH])
        self.b_conv2 = self.bias_variable('conv2_bias', [CONV2_DEPTH])
        self.weights[self.W_conv2.name] = self.W_conv2
        self.weights[self.b_conv2.name] = self.b_conv2

        # This isfollowed by a third convolutional layer that convolves 64 filters of 3 x 3
        # with stride 1 followed by a rectifier
        CONV3_DEPTH = 64
        self.W_conv3 = self.weight_conv_variable("conv3", [3, 3, CONV2_DEPTH, CONV3_DEPTH])
        self.b_conv3 = self.bias_variable('conv3_bias', [CONV3_DEPTH])
        self.weights[self.W_conv3.name] = self.W_conv3
        self.weights[self.b_conv3.name] = self.b_conv3

        FC1_SIZE = 512
        self.W_fc1 = self.weight_variable("fc1", [7 * 7 * CONV3_DEPTH, FC1_SIZE])
        self.b_fc1 = self.bias_variable('fc1_bias', [FC1_SIZE])
        self.weights[self.W_fc1.name] = self.W_fc1
        self.weights[self.b_fc1.name] = self.b_fc1

        self.out_layer = self.weight_variable("readself.out_layer", [FC1_SIZE, self.outpt_size])
        self.bias_layer = self.bias_variable('readself.out_layer_bias', [self.outpt_size])
        self.weights[self.out_layer.name] = self.out_layer
        self.weights[self.bias_layer.name] = self.bias_layer
        print "Model variables created."

    def model(self):

        # convolves 32 8×8 filters with stride 4
        h_conv1 = tf.nn.relu(self.conv2d(self._input, self.W_conv1, strides=[1, 4, 4, 1], padding='VALID') + self.b_conv1)
        # h_pool1 = self.max_pool(h_conv1, strides=[1, 4, 4, 1])
        #print h_conv1

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv2)
        # h_pool2 = self.max_pool(h_conv2, strides=[1, 2, 2, 1])
        #print h_conv2

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID') + self.b_conv3)
        # h_pool3 = self.max_pool(h_conv3, strides=[1, 1, 1, 1])
        #print h_conv3

        shape = h_conv3.get_shape().as_list()
        h_pool_flat = tf.reshape(h_conv3, [-1, shape[1] * shape[2] * shape[3]])

        # First fully connected layer
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, self.W_fc1) + self.b_fc1)
        q_actions = tf.matmul(h_fc1, self.out_layer) + self.bias_layer

        return q_actions