import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        # put None as the last because if more than one action has equal valus in the Q table it will always pick the first one - thus always prioritizing some movement in this scenario
        self.possible_actions = ['forward', 'left', 'right', None];
        self.init_QTable()
        self.initial_learning_rate = 1.0
        self.discount_factor = 0.7
        self.time_step = 1.0
        self.initial_rand_action_prob = 0.2

        random.seed(999)

    def print_QTable(self):
        for state, values in self.QTable.iteritems():
            print state, values

    def init_QTable(self):
        self.QTable = {}

    def get_QValue(self, state, action):
        return self.QTable[state][self.get_action_index(action)]

    def get_action_index(self, action):
        for idx, act in enumerate(self.possible_actions):
            if action == act:
                return idx

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.init_QTable()
        self.time_step = 1.0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = self.get_current_state_based_on_input(inputs)

        # ensure that all needed states are present in the QTable
        if self.state not in self.QTable:
            self.QTable[self.state] = np.array([0.,0.,0.,0.])

        # TODO: Select action according to your policy
        action = self.select_action_according_to_policy()
        #action = self.get_random_action()
        #action = action = self.possible_actions[np.argmax(self.QTable[self.state])]

        # Execute action and get reward
        reward = self.env.act(self, action)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # TODO: Learn policy based on state, action, reward
        self.updateQValue(action, reward)

    def exponential_decay(self, learning_rate, global_step, decay_steps, decay_rate):
        decayed_learning_rate = learning_rate * decay_rate ** (global_step / decay_steps)
        return decayed_learning_rate

    def updateQValue(self, action, reward):
        previous_state = self.state

        inputs = self.env.sense(self)
        current_state = self.get_current_state_based_on_input(inputs)

        if current_state not in self.QTable:
            self.QTable[current_state] =  np.array([0.,0.,0.,0.])

        alpha = self.initial_learning_rate / self.time_step
        self.time_step += 1

        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        self.QTable[previous_state][self.get_action_index(action)] = (1 - alpha) * self.QTable[previous_state][self.get_action_index(action)] + alpha * (reward + self.discount_factor * np.max(self.QTable[current_state]))

        # sprint self.print_QTable()

        # print "State: {}, Action: {}, Reward: {}, Current State: {}".format(previous_state, action, reward, current_state)

    def get_next_move(self):
        return np.argmax(self.QTable[self.state])

    def get_current_state_based_on_input(self, inputs):
        return 'light: {}, left: {}, oncoming: {}, right {}, nextwaypoint: {}'.format(inputs['light'],
                inputs['left'],
                inputs['oncoming'],
                inputs['right'],
                self.next_waypoint)


    # Take some random actions based on probability
    def select_action_according_to_policy(self):
        prob_rate_decay = self.exponential_decay(self.initial_rand_action_prob, self.time_step, decay_steps=5, decay_rate=0.84)
        if (random.random() < prob_rate_decay):
            print "Probability rate decay: ", prob_rate_decay
            return self.get_random_action()
        else:
            action = self.possible_actions[np.argmax(self.QTable[self.state])]
            print "Q learning policy action: ", action
            return action

    def get_random_action(self):
        action = self.possible_actions[random.randrange(0, 4, 1)]
        return action


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "Percentage completed: ", e.completed_trials / 100.0

if __name__ == '__main__':
    run()
