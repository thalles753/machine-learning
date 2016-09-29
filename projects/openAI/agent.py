import pdb
import matplotlib.pyplot as plt

class GameBoard:
    def __init__(self):
        self.x_offset = 4
        self.y_offset = 2
        self.cell_height = 12
        self.cell_width = 8
        self.grid_width = 19
        self.grid_height = 14

    def get_cell(self, observation, row, col):
        """
        Gets a sub image from the image matrix in accordance with
        the game grid
        """
        x = self.x_offset + self.cell_width*col
        y = self.y_offset + self.cell_height*row
        print x, y
        return observation[y:(y+self.cell_height), x:(x+self.cell_width)]

    def draw_grid_by_cell(self, observation):
        # Four axes, returned as a 2-d array
        # f, axarr = plt.subplots(self.grid_height, self.grid_width)
        fig = plt.figure()
        img_id = 1
        for row in range(0,self.grid_height):
            for col in range(0,self.grid_width):
                plt.subplot(self.grid_height, self.grid_width,img_id)
                cell = self.get_cell(observation, row, col)
                plt.imshow(cell)
                img_id += 1
                plt.axis('off')
                # axarr[row, col].plot(cell[:,:,1])
        plt.show()

class LearningAgent:
    def __init__(self, env):
        self._env = env
        self._observation = self.reset_env()
        self._action_meanings = self._env.get_action_meanings()
        self.board = GameBoard()
        self.board.draw_grid_by_cell(self._observation)

    def reset_env(self):
        return self._env.reset()

    def get_random_action(self):
        return self._env.action_space.sample()

    def get_action_name(self, action):
        return self._action_meanings[action]

    def update(self, step):
        print ("Step:", step)
        self._env.render()

        # get action
        action = self.get_random_action()
        print "Action: ", self.get_action_name(action)

        # execute action
        self._observation, reward, done, info = self._env.step(action)

        # pdb.set_trace()
        #cell_block = self.board.get_cell(self._observation, 0, 1)
        #plt.imshow(cell_block)
        #plt.show()

    	# print "Reward:", reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            return
