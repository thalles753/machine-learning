import gym
import tensorflow as tf
import DQN
import utils

IMAGE_SIZE=84
AGENT_HISTORY_LENGTH=4

GAME_NAME= "DemonAttack-v0" # "MsPacman-v0"
env = gym.make(GAME_NAME)

session = tf.Session()

network_input_shape = (IMAGE_SIZE, IMAGE_SIZE, AGENT_HISTORY_LENGTH)
with tf.variable_scope("test") as test_scope:
    test_network = DQN.Network(network_input_shape, env.action_space.n)

session.run(tf.initialize_all_variables())

saver = tf.train.Saver()
# Save the variables to disk.
saver.restore(session, "./model/model.ckpt")
print("Model restored.")