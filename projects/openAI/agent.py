import pdb

class GameBoard:
    def __init__(self):
        self.n_rows = 0
        self.n_cols = 0


class LearningAgent:
    def __init__(self, env):
        self._env = env
        self._observation = self.reset_env()
        self._action_meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

    def reset_env(self):
        return self._env.reset()

    def get_random_action(self):
        return self._env.action_space.sample()

    def get_action_name(self, action):
        return self._action_meanings[action]

    def update(self, step):
        print ("Step:", step)
        self._env.render()

        pdb.set_trace()

        # get action
        action = self.get_random_action()
        print "Action: ", self.get_action_name(action)

        # execute action
        self._observation, reward, done, info = self._env.step(action)
    	# print "Reward:", reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            return
