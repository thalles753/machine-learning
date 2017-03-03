import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# debug routine
def check_prev_post_state(prev, post):
    a = np.array_equal(prev[:, :, 1], post[:, :, 0])
    b = np.array_equal(prev[:, :, 2], post[:, :, 1])
    c = np.array_equal(prev[:, :, 3], post[:, :, 2])

    if (not a or not b or not c):
        raise "The post state does not comply with the previous state."
    return True


# debug routine
def display_transition(actions_names, memory):
    # display post states
    states = [memory[0], memory[3]]
    id = 0
    f, axarr = plt.subplots(2, 4, figsize=(18,8))
    for state in states:
        for c in range(0,4):
            axarr[id][c].imshow(state[:,:,c], cmap=plt.cm.Greys);
            axarr[id][c].set_title('Action: ' + actions_names[memory[1]] + " Reward:" + str(memory[2]))
        id += 1
    plt.show()


def copy_model_parameters(sess, from_net, to_net):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      from_net: Estimator to copy the paramters from
      to_net: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(from_net.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(to_net.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)