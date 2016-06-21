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
        self.possible_actions = [None, 'forward', 'left', 'right'];
        self.valid_states = self.get_states()
        self.init_QTable()
        self.alpha = 1.0
        self.time_step = 1.0
        self.random_action_prob = 1.0
        random.seed(999)

    def print_QTable(self):
        for state, values in self.QTable.iteritems():
            print state, values

    def init_QTable(self):
        self.QTable = {}
        #for state in self.valid_states:
        #    self.QTable[state] = np.array([0., 0., 0., 0.])

    def get_QValue(self, state, action):
        return self.QTable[state][self.get_action_index(action)]

    def get_action_index(self, action):
        for idx, act in enumerate(self.possible_actions):
            if action == act:
                return idx

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # self.init_QTable()
        # self.alpha = 1.0
        # self.time_step

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = self.get_current_state_based_on_input(inputs)
        print "Departing from: ", self.state

        if self.state not in self.QTable:
            self.QTable[self.state] =  np.array([0.,0.,0.,0.])

        # TODO: Select action according to your policy
        if (random.random() < self.random_action_prob):
            print "Random action"
            action = self.possible_actions[random.randrange(0, 4, 1)]
        else:
            print "Q action: ", np.argmax(self.QTable[self.state])
            action = self.get_action_index(np.argmax(self.QTable[self.state]))
            print "Move to: ", action
        # Execute action and get reward
        reward = self.env.act(self, action)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # TODO: Learn policy based on state, action, reward
        self.updateQValue(action, reward)

        self.time_step += 1
        self.alpha = self.alpha / self.time_step

        if self.time_step % 20 == 0 and self.random_action_prob > 0.2:
            self.random_action_prob -= 0.1


    def updateQValue(self, action, reward):
        previous_state = self.state

        inputs = self.env.sense(self)
        current_state = self.get_current_state_based_on_input(inputs)
        print "Current state: ", current_state
        #print "Previous State:", previous_state, "QValue:", self.get_QValue(previous_state, action)

        if current_state not in self.QTable:
            self.QTable[current_state] =  np.array([0.,0.,0.,0.])

        print "Alpha: ", self.alpha

        # Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        self.QTable[previous_state][self.get_action_index(action)] += self.alpha * (reward + np.max(self.QTable[current_state] - self.get_QValue(previous_state, action)))

        #print "Current State:", current_state, "QValue:", self.get_QValue(previous_state, action)

        print self.print_QTable()

        #print "State: {}, Action: {}, Reward: {}, Current State: {}".format(previous_state, action, reward, current_state)

    def get_next_move(self):
        return np.argmax(self.QTable[self.state])


    def get_current_state_based_on_input(self, inputs):
        current_state = inputs["light"] + "_light"

        if inputs["oncoming"] != None:
            current_state += "_oncoming" + "_going_" + str(inputs["oncoming"])

        elif inputs["right"] != None:
            current_state += "_left" + "_going_" + str(inputs["right"])

        elif inputs["left"] != None:
            current_state += "_left" + "_going_" + str(inputs["left"])

        return current_state

    def get_states(self):
        lights = ["green_light", "red_light"]
        directions = ['oncoming', 'right', 'left']
        destinations = ['forward', 'left', 'right']
        states = []
        for l in lights:
            for d in directions:
                for dest in destinations:
                    states.append(l + "_" + d + "_going_" + dest)
        states.append(lights[0])
        states.append(lights[1])
        return states

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
