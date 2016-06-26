import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, learning_rate, discount_factor, greedy_policy):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        # put None as the last because if more than one action has equal valus in the Q table it will always pick the first one - thus always prioritizing some movement in this scenario
        self.possible_actions = ['forward', 'left', 'right', None];
        self.init_QTable()
        self.initial_learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_greedy_policy = greedy_policy
        self.completed_trials = 0

        random.seed(999)

    def print_QTable(self):
        for state, values in self.QTable.iteritems():
            print state, values

    def init_QTable(self):
        self.QTable = {}

    # get the QValue for that state/action
    def get_QValue(self, state, action):
        return self.QTable[state][self.get_action_index(action)]

    # given an action return its index
    def get_action_index(self, action):
        return self.possible_actions.index(action)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        #self.init_QTable()

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
        #action = random.choice(self.possible_actions)
        #action = action = self.possible_actions[np.argmax(self.QTable[self.state])]
        action = self.select_action_according_to_policy(t+1)

        # Execute action and get reward
        reward = self.env.act(self, action)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # TODO: Learn policy based on state, action, reward
        self.updateQValue(action, reward, t+1)

        # print self.print_QTable()

        # check if the agent has reached the destination with positive reward
        if self.env.trial_data['success'] == 1:
            self.completed_trials += 1


    # perform an exponential decay method
    def exponential_decay(self, initial_rate, global_step, decay_steps, decay_rate, staircase=False):
        div = global_step / decay_steps
        if staircase == True:
            div = int(global_step / decay_steps)

        decayed_learning_rate = initial_rate * decay_rate ** div
        return decayed_learning_rate

    # perform the Q-Learning equation for each state/action
    def updateQValue(self, action, reward, time_step):
        previous_state = self.state

        # get the new current state
        inputs = self.env.sense(self)
        current_state = self.get_current_state_based_on_input(inputs)

        # if not there, add this state to the QTable
        if current_state not in self.QTable:
            self.QTable[current_state] = np.array([0.,0.,0.,0.])

        alpha = self.initial_learning_rate / time_step

        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        self.QTable[previous_state][self.get_action_index(action)] = (1.0 - alpha) * self.QTable[previous_state][self.get_action_index(action)] + alpha * (reward + self.discount_factor * np.max(self.QTable[current_state]))

        print self.print_QTable()
        # print "State: {}, Action: {}, Reward: {}, Current State: {}".format(previous_state, action, reward, current_state)

    # generate a state string based on the current input
    def get_current_state_based_on_input(self, inputs):
        return 'light: {}, oncoming {}, left: {}, nextwaypoint: {}'.format(inputs['light'],
                inputs['oncoming'], inputs['left'],
                self.next_waypoint)

    # policy that defines the action selection
    def select_action_according_to_policy(self, time_step):
        epsilon = self.exponential_decay(self.initial_greedy_policy, time_step, decay_steps=1, decay_rate=0.86)
        if random.random() < epsilon:
            action = self.possible_actions[random.randint(0, 3)]
            print "------ [Exploration] ------- Choosing random action:", action
        else:
            q_row = list(self.QTable[self.state])
            max_q_value = max(q_row)
            count = q_row.count(max_q_value)
            if count > 1: # check for more than one max value
                best_options = []
                for action_index in range(len(self.possible_actions)-1): # exclude the None action
                    if q_row[action_index] == max_q_value:
                        best_options.append(action_index)
                action_index = random.choice(best_options)
                # print "[Exploitation/Exploration] Choosing best action among:", best_options, "Selected:", self.possible_actions[action_index]
            else:
                action_index = q_row.index(max_q_value)
                # print "[Exploitation] Best action:", self.possible_actions[action_index]
            action = self.possible_actions[action_index]
        return action

def run():
    f = open('running_report.txt', 'w')

    # setup various parameter combinations
    discount_factors = [0.5]
    starting_learning_rates = [0.5]
    epsilon_greedy_policy = [0.09]

    for d_factor in discount_factors:
        for alpha in starting_learning_rates:
            for greedy_policy in epsilon_greedy_policy:

                """Run the agent for a finite number of trials."""
                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, learning_rate=alpha, discount_factor=d_factor, greedy_policy=greedy_policy)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0, display=True)  # create simulator (uses pygame when display=True, if available)

                number_of_trials = 100

                # NOTE: To speed up simulation, reduce update_delay and/or set display=False
                sim.run(n_trials=number_of_trials)  # run for a specified number of trials

                #NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                print >> f, "Learning rate:", alpha
                print >> f, "Discount factor:", d_factor
                print >> f, "Greedy Policy:", greedy_policy
                print >> f, "Percentage completed: ", a.completed_trials / 100.0, "\n"

                f.flush()
    f.close()

if __name__ == '__main__':
    run()
