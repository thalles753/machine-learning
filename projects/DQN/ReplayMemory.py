import random
import numpy as np

class ReplayMemory:
    def __init__(self, args):
        self.SIZE = args.replay_memory_size
        self.AGENT_HISTORY_LENGTH = args.agent_history_length
        self.MINIBATCH_SIZE = args.minibatch_size
        self.IMAGE_SIZE = args.screen_width

        self.head_index = -1;
        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminals = None

    def size(self):
        return len(self.observations)

    def init_memory(self, obs):
        self.observations = np.stack(tuple(obs for _ in range(self.AGENT_HISTORY_LENGTH)), axis=2)
        self.actions = np.array([0,0,0,0])
        self.rewards = np.array([0,0,0,0])
        self.terminals = np.array([False,False,False,False])
        assert self.observations.shape[2] == len(self.actions) == len(self.rewards) == len(self.terminals), "Invalid size"
        self.head_index = 3;

    def pop(self):
        self.observations = np.delete(self.observations, 0, axis=2)
        self.actions = np.delete(self.actions, 0)
        self.rewards = np.delete(self.rewards, 0)
        self.terminals = np.delete(self.terminals, 0)
        self.head_index -= 1

    def add(self, obs, action, reward, terminal):
        if self.head_index == (self.SIZE - 1):
            self.pop()

        assert (obs.shape == (self.IMAGE_SIZE, self.IMAGE_SIZE)), "Image shape not expected, please provide an (" + str(self.IMAGE_SIZE) + ", " + str(self.IMAGE_SIZE) + ") image."
        assert self.observations.shape[2] == len(self.actions) == len(self.rewards) == len(self.terminals), "Invalid size"
        obs = np.expand_dims(obs, axis=2)
        self.observations = np.append(self.observations, obs, axis=2)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)
        self.terminals = np.append(self.terminals, terminal)
        self.head_index += 1

    def _get_state(self, index):
        state = self.observations[:,:,index - 3 : index + 1]
        print state.shape
        assert state.shape == (84,84,4), "State shape error!"
        return state

    def get_current_state(self):
        print self.observations.shape
        return self.observations[:,:,self.head_index - 3 : self.head_index + 1]

    def get_current_transition(self):
        minibatch = []
        pre_state = self._get_state(self.head_index - 1)
        post_state = self._get_state(self.head_index)
        minibatch.append([pre_state, self.actions[self.head_index], self.rewards[self.head_index], post_state, self.terminals[self.head_index]])
        return minibatch

    def get_minibatch(self):
        indexes = random.sample(range(self.AGENT_HISTORY_LENGTH, len(self.actions)), self.MINIBATCH_SIZE)
        print indexes
        minibatch = []
        for index in indexes:
            pre_state = self._get_state(index - 1)
            post_state = self._get_state(index)
            minibatch.append([pre_state, self.actions[index], self.rewards[index], post_state, self.terminals[index]])
        return minibatch