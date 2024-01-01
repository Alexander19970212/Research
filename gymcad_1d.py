import numpy as np
import torch
import random

def find_indeces(coords, space_size, state_size):
    scale = state_size/space_size
    index_list = []
    for line in coords:
        indeces = [int(line[0]*scale), int(line[1]*scale)]
        index_list.append(indeces)

    return index_list

class BoxCad_1d:
    """Box Cad 1d environment.
  box CAD is a simplified CAD 1d environment for agent training.
  When the environment is reseted, env creates a task for the agent:
    1) updates the space (removes all geometry from the previous episode).
    2) creates lines-tasks in space that the body of the construction must reach;
    3) creates the current working line

  Available operations with the construction line:
  - create a new line (in the future !!!!)
  - independently move the points of working line.

  The oservation is a pytorch tensor, interpreted as a geometric space.
  It looks like |000000111111100000|
  two options are available:
  1) self.multichannel = False. Paytorch tensor size (2, state_size). 
    - there are three one-dimensional spaces. The first space is the space of details, 
    the value of the elements in the tensor: 
      1 - if there is the current working line;
      0 - if not. 
    The second space is the task space, the value of the tensor elements: 
      1 - if there is a task body, 
      0 - if not. 
  2) self.mutechannel = True. Paytorch tensor size (state_size). 
    All bodies are in one space at once, the values of the elements
    are equal to the sum of the elements from the previous case through 
    weighting coefficients.

  Actions:
  full list of actions - range(0, 4).
    0 - move the right point of working line to the right;
    1 - move the right point of working line to the left;
    2 - move the left point of working line to the right;
    3 - move the left point of working lone to the left.

  Rewards. 
    1 if working line achieved the task point;
    0 if doesn't;
    -1 if the task point is inside the working line.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = 32
        self.space_size = 100
        self.moving_step = 10
        self.max_index = 100

        self.mesh_space = np.arange(0, self.space_size+self.moving_step, self.moving_step)

        self.multichannel = True

        if self.multichannel:
            self.working_space = torch.zeros(2, self.state_size).to(self.device)

        else:
            self.working_space = torch.zeros(self.state_size).to(self.device)

        self.task_lines = None
        self.current_line = torch.tensor([0, 0]).to(self.device)

        self.workspace_tasks = torch.zeros(self.state_size).to(self.device)
        self.workspace_current_line = torch.zeros(self.state_size).to(self.device)

        self.touched_points_indeces = []

        self.current_objective = 0
        self.action = None

        self.generate_tasks()
        self.generate_working_line()
        self.calculate_objective()

        self.zero_length_flag = False
        self.done_flag = False
        self.current_index = 0


    def reset(self):
        if self.multichannel:
            self.working_space = torch.zeros(2, self.state_size).to(self.device)

        else:
            self.working_space = torch.zeros(self.state_size).to(self.device)

        self.task_lines = None
        self.current_line = torch.tensor([0, 0]).to(self.device)

        self.workspace_tasks = torch.zeros(self.state_size).to(self.device)
        self.workspace_current_line = torch.zeros(self.state_size).to(self.device)

        self.current_objective = 0
        self.touched_points_indeces = []
        self.action = None

        self.generate_tasks()
        self.generate_working_line()
        self.calculate_objective()

        self.zero_length_flag = False
        self.done_flag = False
        self.current_index = 0 

    def check_ability_to_touch(self):
        pass

    def update_workspace_task(self):
        self.workspace_tasks *= 0
        indeces = find_indeces(self.task_lines, self.space_size, self.state_size) # [[x_11, x_12], [x_21, x_22]]
        for line in indeces:
            self.workspace_tasks[line[0]:line[1]] = 1

    def update_workspace_current_line(self):
        self.workspace_current_line *= 0
        indeces = find_indeces(self.current_line[None, :], self.space_size, self.state_size)
        self.workspace_current_line[indeces[0][0]:indeces[0][1]] = 1

    def generate_tasks(self, mesh_size=10, offset=20):
        # define the number of boxes
        number_of_task_point = random.sample([1, 2], k=1)[0]

        # self.task_lines = torch.tensor(number_of_task_point, 2).to(self.device)
        # generate ranodm points in the first third of space and withing the last third of space
        working_space_first_point = self.mesh_space[self.mesh_space<self.space_size/3]
        working_space_second_point = self.mesh_space[self.mesh_space>self.space_size*2/3]

        first_point = np.random.choice(working_space_first_point, 1, False)[0]
        second_point = np.random.choice(working_space_second_point, 1, False)[0]

        if number_of_task_point == 1:
            point = random.sample([first_point, second_point], k=1)[0]
            
            # generate offset
            if point >= self.space_size/2:
                self.task_lines = torch.tensor([[point, point+offset]]).to(self.device)
                
            else:
                self.task_lines = torch.tensor([[point-offset, point]]).to(self.device)
                
        else:
            # generate offset
            self.task_lines = torch.tensor([[first_point-offset, first_point], [second_point, second_point+offset]]).to(self.device) 

        # cut if the second point out of space
        self.task_lines = torch.clip(self.task_lines, 0, self.space_size)

        self.update_workspace_task()
        # update workspace task

    def generate_working_line(self, mesh_size=10):
        # select two points within the second third of space
        working_space = self.mesh_space[self.mesh_space>self.space_size/3]
        working_space = working_space[working_space<self.space_size*2/3]
        working_points = np.random.choice(working_space, 2, False)

        # sort them
        working_points = np.sort(working_points)

        # update parameters
        self.current_line[0] = working_points[0]
        self.current_line[1] = working_points[1]

        self.update_workspace_current_line()

    def transform_working_line(self, moving_action):
        """
        Actions:

            0 - move the right point of working line to the right;
            1 - move the right point of working line to the left;
            2 - move the left point of working line to the right;
            3 - move the left point of working lone to the left.

        """
        # decode moving case
        move_parametes = [[1, 1], [1, -1], [0, 1], [0, -1]]
        moving_case = move_parametes[moving_action]

        # transforme current line
        new_line = self.current_line.copy()
        new_line[moving_case[0]] = new_line[moving_case[0]] + self.moving_step*moving_case[1]

        # update if new line is not zero length 
        # !!!!!!!!!!!!May be it is necessary to update the current line anyway
        if new_line[0] != new_line[1]:
            self.current_line = new_line
            self.update_workspace_current_line()
            self.zero_length_flag = False

        else:
            self.zero_length_flage = True

        self.calculate_objective()

    def calculate_objective(self):
        ### For the policy method from homework the best result is zero
        # so we will return the sum of distant to the task point and doubled overlaping length

        # calulate the distant to the task points
        dist = 0

        if self.task_lines.size(dim=0) == 2:

            dist_1 = self.current_line[0] - self.task_lines[0, 1]
            if dist_1 < 0: # overlap
                dist_1 = dist_1 * (-2)
            dist_2 = self.task_lines[1, 0] - self.current_line[1]
            if dist_2 < 0: #overlap
                dist_2 = dist_2 * (-2)
            dist = dist_1+dist_2

        else:
            if self.task_lines[0, 0] < self.space_size/2:
                dist = self.current_line[0] - self.task_lines[0, 1]
            else:
                dist = self.task_lines[0, 0] - self.current_line[1]

            if dist < 0:
                dist = dist * (-2)

        if dist == 0:
            self.done_flag = True

        self.current_objective = dist

    def update_workspace(self):

        if self.multichannel:
            # self.working_space = torch.zeros(2, self.state_size).to(self.device)
            self.working_space[0] = self.workspace_current_line
            self.working_space[1] = self.workspace_tasks

        else:
            # self.working_space = torch.zeros(self.state_size).to(self.device)
            self.working_space *= 0
            self.working_space = self.workispace_tasks + 0.5 * self.workspace_current_line

    def receive_action(self, action):
        self.action = action

    def step(self):

        if self.done_flag == False and self.current_index < self.max_index:
            self.transform_working_line(self.action)
            self.current_index += 1
            return True
        
        else: return False

    def get_sim_step_data(self):
        
        return (self.working_space, self.action, self.current_index)

    def render(self):
        print(self.working_space)


















    


    



