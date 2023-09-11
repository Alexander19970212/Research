import torch

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from matplotlib import colors

import random

import numpy as np


def find_indexes(coords, space_size, state_size, sorting=True):
  """The function finds the indices of points in a space matrix 
  as if the matrix represented a three-dimensional cubic 
  space of the specified size with the origin at the center 
  and the points given three coordinates.
  Inputs:
  coors      - pytorch tensor, size = 2, 3 (2 points with 3 coords)
  space size - num,            size of 3D space
  state size - int,            size of space matrix
  sorting    - bool,           if it is necessary to return indexes in positive direction
  Rutern:
    index related to the first point and x-axis,  int
    index related to the first point and y-axis,  int
    index related to the first point and z-axis,  int
    index related to the second point and x-axis, int
    index related to the second point and y-axis, int
    index related to the second point and z-axis, int
  """


  indeces = (coords + 0.5*space_size)*state_size/space_size
  indeces = torch.round(indeces)

  if sorting:
    indeces, _ = torch.sort(indeces, 0)

  return int(indeces[0, 0]), int(indeces[0, 1]), int(indeces[0, 2]), int(indeces[1, 0]), int(indeces[1, 1]), int(indeces[1, 2])

def generate_normal_distribution_list(start_x, end_x):

  mu, sigma = 0, 10

  range_indeces = np.arange(start_x, end_x)
  distribution_list = ((2*np.pi*sigma**2)**0.5)*np.exp((-1)*((range_indeces-mu)**2)/(2*sigma**2))

  return range_indeces, distribution_list

def generate_random_box2(space_size, half_edge, offset_rate=0.03, thikness_rate=0.1):
  
  x, y, z = half_edge, half_edge, half_edge
  all_possible_faces = np.array([[[-x, -y, z], [-x, y, z], [x, y, z], [x, -y, z]], 
                        [[-x, -y, -z], [-x, y, -z], [x, y, -z], [x, -y, -z]], 
                        [[x, -y, -z], [x, -y, z], [x, y, z], [x, y, -z]],      #
                        [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, z]],
                        [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]],
                        [[-x, -y, -z], [-x, -y, z], [x, -y, z], [x, -y, z]]
                        ])

  offsets_mask = np.array([[0, 0, 1],
             [0, 0, -1],
             [1, 0, 0],
             [-1, 0, 0],
             [0, 1, 0],
             [0, -1, 0]])

  face_index = random.choice(range(6))

  face = all_possible_faces[face_index] + offsets_mask[face_index]*int(offset_rate*space_size)
  face = torch.tensor(face)

  box_coords = torch.zeros(2, 3)
  box_coords[0] = face[0]
  box_coords[1] = face[2] + offsets_mask[face_index]*thikness_rate*space_size

  return  face, torch.tensor(offsets_mask[face_index]), box_coords


def generate_random_box(space_size, scale_step, reverse_distr = True, max_size_rate=0.6, min_size_rate=0.4, thikness_rate=0.3):

  size_range = np.arange(round(space_size*min_size_rate), round(space_size*max_size_rate))

  thikness = space_size*thikness_rate

  # generate random parameters

  Normal_X_axis_Y_axis = random.sample([0, 1, 2], 3)
  Normal = Normal_X_axis_Y_axis[0]
  X_axis = Normal_X_axis_Y_axis[1]
  Y_axis = Normal_X_axis_Y_axis[2]

  list_coords, weights = generate_normal_distribution_list(-space_size*0.5, space_size*0.5)
  if reverse_distr:
    weights = 1 - weights/np.max(weights)*0.8

  else:
    weights = weights/np.max(weights)*0.9

  x = random.choices(list_coords, weights=weights, k=1)[0]
  y = random.choices(list_coords, weights=weights, k=1)[0]
  z = random.choices(list_coords, weights=weights, k=1)[0]

  x_size = random.choice(size_range)
  y_size = random.choice(size_range)

  # prepare face coords

  face_coords = torch.zeros(4, 3)

  face_coords[:, 0] = x_size
  face_coords[:, 1] = y_size
  face_coords[:, 2] = 0

  face_mask = torch.tensor([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])

  face_coords = face_coords*face_mask

  center_coords = torch.tensor([x, y, z])

  inter_face_coords = torch.zeros(4, 3)

  inter_face_coords[:, X_axis] = face_coords[:, 0]
  inter_face_coords[:, Y_axis] = face_coords[:, 1]
  inter_face_coords[:, Normal] = face_coords[:, 2]

  face_coords = inter_face_coords + center_coords

  # prepare normal vector

  center_coords = np.array([x, y, z])

  direction = (center_coords[Normal] < 0)*2 - 1
  normal_vector = torch.tensor([0, 0, 0])
  normal_vector[Normal] = direction

  # prepare box

  box_coords = torch.zeros(2, 3)
  box_coords[0] = face_coords[0]
  box_coords[1] = face_coords[2]

  box_coords[1, Normal] = box_coords[1, Normal] - direction*thikness
  box_coords, _ = torch.sort(box_coords, 0)

  # procces the out of size

  outing_mask_face_pos =  (face_coords > space_size*0.5)*1
  outing_mask_face_neg =  (face_coords < -space_size*0.5)*1

  face_coords = face_coords * (1-outing_mask_face_pos) + outing_mask_face_pos*0.5*space_size
  face_coords = face_coords * (1-outing_mask_face_neg) - outing_mask_face_neg*0.5*space_size

  outing_mask_box_pos = (box_coords > 0.5*space_size)*1
  outing_mask_box_neg = (box_coords < -0.5*space_size)*1

  box_coords = box_coords * (1 - outing_mask_box_pos) + outing_mask_box_pos*0.5*space_size
  box_coords = box_coords * (1 - outing_mask_box_neg) - outing_mask_box_neg*0.5*space_size

  face_coords = (face_coords//scale_step)*scale_step
  box_coords = (box_coords//scale_step)*scale_step



  return face_coords, normal_vector, box_coords

def check_intersections_two_boxes(box_1, box_2):
  min_x1 = box_1[0, 0]
  min_x2 = box_2[0, 0]
  max_x1 = box_1[1, 0]
  max_x2 = box_2[1, 0]

  min_y1 = box_1[0, 1]
  min_y2 = box_2[0, 1]
  max_y1 = box_1[1, 1]
  max_y2 = box_2[1, 1]

  min_z1 = box_1[0, 2]
  min_z2 = box_2[0, 2]
  max_z1 = box_1[1, 2]
  max_z2 = box_2[1, 2]

  a_x_face1 = abs(max_x1 - min_x1)/2
  a_y_face1 = abs(max_y1 - min_y1)/2
  a_z_face1 = abs(max_z1 - min_z1)/2

  a_x_face2 = abs(max_x2 - min_x2)/2
  a_y_face2 = abs(max_y2 - min_y2)/2
  a_z_face2 = abs(max_z2 - min_z2)/2  

  x_center_face1 = min_x1 + (max_x1 - min_x1)/2
  y_center_face1 = min_y1 + (max_y1 - min_y1)/2
  z_center_face1 = min_z1 + (max_z1 - min_z1)/2

  x_center_face2 = min_x2 + (max_x2 - min_x2)/2
  y_center_face2 = min_y2 + (max_y2 - min_y2)/2
  z_center_face2 = min_z2 + (max_z2 - min_z2)/2

  distance_x = abs(x_center_face1 - x_center_face2)
  distance_y = abs(y_center_face1 - y_center_face2)
  distance_z = abs(z_center_face1 - z_center_face2)

  rects_x_toughts = (a_x_face1 + a_x_face2) > distance_x
  rects_y_toughts = (a_y_face1 + a_y_face2) > distance_y
  rects_z_toughts = (a_z_face1 + a_z_face2) > distance_z

  return  rects_x_toughts and rects_y_toughts and rects_z_toughts


def check_intersections_two_boxes2(box_1, box_2):


  min_x1 = box_1[0, 0]
  min_x2 = box_2[0, 0]
  max_x1 = box_1[1, 0]
  max_x2 = box_2[1, 0]

  min_y1 = box_1[0, 1]
  min_y2 = box_2[0, 1]
  max_y1 = box_1[1, 1]
  max_y2 = box_2[1, 1]

  min_z1 = box_1[0, 2]
  min_z2 = box_2[0, 2]
  max_z1 = box_1[1, 2]
  max_z2 = box_2[1, 2]

  eps = 1

  res_1 = (min_x1-eps <= min_x2 and min_x2 <= max_x1+eps) or (min_x2-eps <= min_x1 and min_x1 <= max_x2+eps)
  res_2 = (min_y1-eps <= min_y2 and min_y2 <= max_y1+eps) or (min_y2-eps <= min_y1 and min_y1 <= max_y2+eps)
  res_3 = (min_z1-eps <= min_z2 and min_z2 <= max_z1+eps) or (min_z2-eps <= min_z1 and min_z1 <= max_z2+eps)
  
  
        
  result = res_1 or res_2 or res_3
  
  return result
  
  

def check_intersections(cheking_box, previous_boxes):
  
  for box in previous_boxes:
    if check_intersections_two_boxes(cheking_box, box):
      return True

  return False



def poligones_verts(box_coords):

  x1 = box_coords[0, 0]
  x2 = box_coords[1, 0]
  y1 = box_coords[0, 1]
  y2 = box_coords[1, 1]
  z1 = box_coords[0, 2]
  z2 = box_coords[1, 2]

  vert1 = [x1, y1, z1]
  vert2 = [x1, y1, z2]
  vert3 = [x1, y2, z2]
  vert4 = [x1, y2, z1]
  vert5 = [x2, y1, z1]
  vert6 = [x2, y1, z2]
  vert7 = [x2, y2, z2]
  vert8 = [x2, y2, z1]


  verts = [[vert1, vert2, vert3, vert4],
           [vert5, vert6, vert7, vert8],
           [vert1, vert2, vert6, vert5],
           [vert2, vert3, vert7, vert6],
           [vert3, vert4, vert8, vert7],
           [vert4, vert1, vert5, vert8]]

  return verts

def check_boxes_including(box_1, box_2):

  min_x1 = box_1[0, 0]
  min_x2 = box_2[0, 0]
  max_x1 = box_1[1, 0]
  max_x2 = box_2[1, 0]

  min_y1 = box_1[0, 1]
  min_y2 = box_2[0, 1]
  max_y1 = box_1[1, 1]
  max_y2 = box_2[1, 1]

  min_z1 = box_1[0, 2]
  min_z2 = box_2[0, 2]
  max_z1 = box_1[1, 2]
  max_z2 = box_2[1, 2]

  includings_by_x_12 = (min_x1 <= min_x2) and (max_x2 <= max_x1)
  includings_by_x_21 = (min_x2 <= min_x1) and (max_x1 <= max_x2)

  includings_by_y_12 = (min_y1 <= min_y2) and (max_y2 <= max_y1)
  includings_by_y_21 = (min_y2 <= min_y1) and (max_y1 <= max_y2)

  includings_by_z_12 = (min_z1 <= min_z2) and (max_z2 <= max_z1)
  includings_by_z_21 = (min_z2 <= min_z1) and (max_z1 <= max_z2)

  includings_case_12 = includings_by_x_12 and includings_by_y_12 and includings_by_z_12
  includings_case_21 = includings_by_x_21 and includings_by_y_21 and includings_by_z_21

  return  includings_case_21 or includings_case_12

def check_includings(cheking_box, previous_boxes):
  includings = False
  for box in previous_boxes:
    includings = check_boxes_including(cheking_box, box)

  return includings

def check_complanar_faces(face1, face2):
  f1_p = (face1==face1[0])*1
  normal_coord_mask1 = (f1_p.sum(0)==4)
  f2_p = (face2==face2[0])*1
  normal_coord_mask2 = (f2_p.sum(0)==4)

  comparison = normal_coord_mask1*normal_coord_mask2
  equals_in_normal_coord =  (face1[0]*comparison) == (face2[0]*comparison)*1

  if comparison.sum() == 1 and equals_in_normal_coord.sum() == 3:
    return True, comparison
  else:
    return False, 0

def check_faces_intersection(face1, face2, projected_mask):
  # print(projected_mask.get_device())

  devc = projected_mask.get_device()
  pr_m_inds = torch.tensor([0, 1, 2]).to(devc)[(projected_mask==False)]  

  projected_face2 = torch.zeros(4, 2)
  projected_face1 = torch.zeros(4, 2)

  k_lin_int = 0.5

  try:
    projected_face1[:, 0] = face1[:, pr_m_inds[0]]
    projected_face2[:, 0] = face2[:, pr_m_inds[0]]
    projected_face1[:, 1] = face1[:, pr_m_inds[1]]
    projected_face2[:, 1] = face2[:, pr_m_inds[1]]
  except:
    print(face1, face2)
    print(pr_m_inds, projected_mask)

  max_x1 = projected_face1[:, 0].max()
  min_x1 = projected_face1[:, 0].min()

  max_x2 = projected_face2[:, 0].max()
  min_x2 = projected_face2[:, 0].min()

  max_y1 = projected_face1[:, 1].max()
  min_y1 = projected_face1[:, 1].min()

  max_y2 = projected_face2[:, 1].max()
  min_y2 = projected_face2[:, 1].min()

  a_x_face1 = abs(max_x1 - min_x1)/2
  a_y_face1 = abs(max_y1 - min_y1)/2

  a_x_face2 = abs(max_x2 - min_x2)/2
  a_y_face2 = abs(max_y2 - min_y2)/2  

  x_center_face1 = min_x1 + (max_x1 - min_x1)/2
  y_center_face1 = min_y1 + (max_y1 - min_y1)/2

  x_center_face2 = min_x2 + (max_x2 - min_x2)/2
  y_center_face2 = min_y2 + (max_y2 - min_y2)/2

  distance_x = abs(x_center_face1 - x_center_face2)
  distance_y = abs(y_center_face1 - y_center_face2)

  rects_x_toughts = (a_x_face1 + a_x_face2) > distance_x
  rects_x_overlap = (a_x_face1 + a_x_face2)*0.5 > distance_x

  rects_y_toughts = (a_y_face1 + a_y_face2) > distance_y
  rects_y_overlap = (a_y_face1 + a_y_face2)*0.5 > distance_y

  return  (rects_x_toughts and rects_y_overlap) or (rects_x_overlap and rects_y_toughts) 


def check_touch_with_face(face, box):

  
  box_faces = poligones_verts(box)
  devc = face.get_device()

  for box_face in box_faces:
    checking_face = torch.tensor(box_face).to(devc)
    complanarity, proj_mask = check_complanar_faces(face, checking_face)
    if complanarity:
      if check_faces_intersection(face, checking_face, proj_mask):
        return True

  return False

def remove_rows_by_indeces(tensor_tourch, rows_indeces, device):
  
  positive_indeces = []

  for index in range(tensor_tourch.size(dim=0)):
    if index not in rows_indeces:
      positive_indeces.append(index)

  result_tensor = torch.zeros(len(positive_indeces), 4, 3).to(device)

  for count, i in enumerate(positive_indeces):
    result_tensor[count] = tensor_tourch[i]

  return result_tensor

class BoxCAD:
  """
  Box Cad environment.
  box CAD is a simplified CAD environment for agent training.
  When the environment is reseted, env creates a task for the agent:
    1) updates the space (removes all geometry from the previous episode).
    2) creates cubes-tasks in space that the body of the construction must reach
    3) creates cubes-constraints; their construction by the body must be avoided.

  Available operations with the construction body:
  - create a new body cube
  - independently move the surface of the cube in the direction of the normal or against it.

  The oservation is a pytorch tensor, interpreted as a geometric space.
  two options are available:
  1) self.multichannel = False. Paytorch tensor size (3, state_size, state_size, state_size). 
    - there are three three-dimensional spaces. The first space is the space of details, 
    the value of the elements in the tensor: 
      0.5 for the previous body, 
      0.75 for the current cube. 
    The triple space is the task space, the value of the tensor elements: 
      1 - if there is a task body, 
      0 - if not. 
    The third space is the space of restrictions, the value of the elements: 
      1 - there are restrictions, 
      0 - there are no restrictions.
  2) self.mutechannel = True. Paytorch tensor size (state_size, state_size, state_size). 
    All bodies are in one space at once, the values of the elements
    are equal to the sum of the elements from the previous case through 
    weighting coefficients.

  Actions:
  full list of actions - range(0, 12).
    0 - create a new cube;
    1 - move the wall +X along the normal axis;
    2 - move the wall +X against the normal axis;
    3 - move the wall -X along the normal axis;
    4 - move the wall -X against the normal axis;
    5 - move the wall +Y along the normal axis;
    6 - move the wall +Y against the normal axis;
    7 - move the wall -Y along the normal axis;
    8 - move the wall -Y against the normal axis;
    9 - move the wall +Z along the normal axis;
    10 - move the wall + Z against the normal axis;
    11 - move the wall -Z along the normal axis;
    12 - move the wall -Z against the normal axis;

  Rewards. The essence of the reward system is the sum through weighting 
  coefficients of the following components:
    - the number of times the current cube touches cubes-tasks. 
      In this case, task cubes that were touched in previous times are not taken into account. 
      (in the future the touch area will be taken into account). The weight is 2.
    - The number of intersections of constraint cubes by the current cube. The weight is -1.
    - The number of intersections of task cubes by the current cube. 
      This is done to prevent the current cube from slipping through the task surface. 
      The weight is -0.1.
    - Reward for the first box. This is done to encourage the agent to create a box when 
      the space is empty, rather than trying to move walls that are not there. 
      The weight is 0.5.
    - Number of steps without the first box. The weight is -0.1.
    - penalty if a new box is created inside a previously created body. 
      Works after the third step in a row when the box is inside. The penalty is -0.2.

  """
  def __init__(self, device):
    self.device = device
    self.state_size = 32 # observation tensor size
    self.space_size = 40 # size of geometric space

    # coordinates of two opposite points of the current cube
    self.current_box = torch.tensor([[0, 0, 0], [0, 0, 0]]).to(self.device)
    self.current_touched_face = [] 
    # self.objective = 0 
    self.multichannel = True

    if self.multichannel:
      self.workspace = torch.zeros(3, self.state_size, self.state_size, self.state_size) # Observation workspace
      self.workspace = self.workspace.to(self.device)
    else:
      self.workspace = torch.zeros(self.state_size, self.state_size, self.state_size)
      self.workspace = self.workspace.to(self.device)

    # Create tensors to save geometric data in tensor format
    self.workspace_previous = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device) # tensor for previous cubes
    self.workspace_task = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device) # tensor for cube-tasks
    self.workspace_constraints = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device) # tensor for cube-constraints
    self.workspace_current = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device) # tensor for current cube
    self.previous_boxes = None

    self.multichannel = True

    self.last_succesful_task_face = None # in order not to get reward for the same face

    # args to save gemetric data (they will be created in reset and generate fuctions)
    self.task_boxes = None 
    self.task_faces = None
    self.task_normales = None
    self.task_faces_clone = None

    self.constraints_boxes = None
    
    self.includings_counter = 0 # count the steps when current box inside previous ones
    self.first_box_creating_counter = 0 #count the steps befor first box is created

    self.first_step_done = False # flag of creating the first box
    self.one_task_sucsesfull = False # now it is not used!!!
    
    # data for controlling touching the cube-tasks
    self.touching_faces_indeces = [] 
    self.reward_for_first_box = False
    self.reward_for_first_task_box_intersection = False
    

    # create initial environment
    self.generate_task()
    self.task_faces_clone = self.task_faces.clone()
    self.generate_constraints()
    
  def reset(self):
    """
    Reset the environment
    """
    self.previous_boxes = None
    self.task_boxes = None
    self.task_faces = None
    self.task_normales = None
    self.task_faces_clone = None
    self.last_succesful_task_face = None
    self.touching_faces_indeces = []
    self.first_box_creating_counter = 0
    self.reward_for_first_box = False
    self.reward_for_first_task_box_intersection = False

    self.constraints_boxes = None
    self.workspace_previous = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device)
    self.workspace_task = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device)
    self.workspace_constraints = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device)
    self.workspace_current = torch.zeros(self.state_size, self.state_size, self.state_size).to(self.device)
    self.first_step_done = False
    self.one_task_sucsesfull = False
    self.workspace *= 0
    self.current_box *= 0
    self.current_touched_face = []
    self.generate_task()
    self.task_faces_clone = self.task_faces.clone()


    self.generate_constraints()
    self.update_workspace()

    if self.multichannel:
      return self.workspace
    else:
      return self.workspace[None, :]

  def action_sample(self):
    return random.choice(range(0, 13))

  def step(self, action):

    reward = 0

                #C, f, d -  C - index of face axis (X - 0, Y - 1, Z - 2)
                            # f - direction face according to the axis 
                            #     (1 - along the axis, 0 - against the axis)
                            # d - direction of movement
                            #     (1 - alonge the normal, -1 - againts the normal)    
    face_dir = [[0, 1, 1],  # +X +1
                [0, 1, -1], # +X -1
                [0, 0, 1],  # -X +1
                [0, 0, -1], # -X -1
                [1, 1, 1],  # +Y +1
                [1, 1, -1], # +Y -1
                [1, 0, 1],  # -Y +1
                [1, 0, -1], # -Y -1
                [2, 1, 1],  # +Z +1
                [2, 1, -1], # +Z -1
                [2, 0, 1],  # -Z +1
                [2, 0, -1]]  # -Z -1


    self.reward_for_first_box = False

    # adding a new box action
    if action == 0:
      if self.first_step_done == False:
        self.reward_for_first_box = True
      self.add_new_box()
      
    # movement of box face
    # it works only in case if the first box is done
    else:
      # face = face_dir[action-1][0]
      # direction = face_dir[action-1][1]

      if self.first_step_done:
        self.move_face(face_dir[action-1])
    
    self.update_workspace()

    reward = self.calculate_objective()

    # the task is done if previous boxes and current box touch all task_face
    done = (list(range(self.task_faces_clone.size(dim=0))) == self.current_touched_face)

    if self.multichannel:
      return self.workspace, reward, done, 0

    else:
      return self.workspace[None, :], reward, done, 0
          

  def add_new_box(self, init_size=0.27):
    """
    Function creates new box.
    Input:
      init_size is scale of new boxes related to the state_size
    """

    self.reward_for_first_task_box_intersection = True
    if self.first_step_done == False:
      self.first_step_done = True

    else:
      # move information from current box tensor to previous boxes tensor  
      self.update_previous_workspace()

      # don't consider task boxes that were achieved
      self.task_faces_clone = remove_rows_by_indeces(self.task_faces_clone, self.current_touched_face, self.device)
      self.current_touched_face = []

    # create standart cube
    self.current_box[0] = int(-0.5*self.space_size*init_size)
    self.current_box[1] = int(0.5*self.space_size*init_size)

    self.update_current_workspace    

  def generate_task(self, minimum_number=2, maximum_number=3):
    """The functions creates [minimun_number, maximum_number] of tasks-boxes. 
    """  

    task_space_scale = 0.4

    # define the number of boxes
    number_of_task_face = random.sample(range(minimum_number, maximum_number+1), k=1)[0]
    
    # Create tensor for saving
    self.task_boxes = torch.zeros(number_of_task_face, 2, 3).to(self.device) # N_boxes, (start-end points), coords 
    self.task_faces = torch.zeros(number_of_task_face, 4, 3).to(self.device) # N_boxes, vertex, coords
    self.task_normales = torch.zeros(number_of_task_face, 3).to(self.device) # N_boxes, vector_coords
    
    # create and save the first box
    face_coords, face_normals, box = generate_random_box2(self.space_size, 5)

    self.task_boxes[0] = box.to(self.device)
    self.task_faces[0] = face_coords.to(self.device)
    self.task_normales[0] = face_normals.to(self.device)

    number_generated_box = 1

    # create other boxes
    while number_generated_box < number_of_task_face:
      face_coords, face_normals, box = generate_random_box2(self.space_size, 5)

      # new box should not intersect previous ones
      if check_intersections(box, self.task_boxes[0:number_generated_box]) == False:
        self.task_boxes[number_generated_box] = box.to(self.device)
        self.task_faces[number_generated_box] = face_coords.to(self.device)
        self.task_normales[number_generated_box] = face_normals.to(self.device)
        number_generated_box += 1

    # update task tensor
    for box in  self.task_boxes:
      x1_index, y1_index, z1_index, x2_index, y2_index, z2_index = find_indexes(box, self.space_size, self.state_size)
      self.workspace_task[x1_index:x2_index, y1_index:y2_index, z1_index:z2_index] = 1


  def generate_constraints(self, min_number_box=1, max_number_box=2):
    """The function creates [min_number_box, max_number_box] number constrain-boxes.
    """

    # define number of boxes
    if max_number_box == min_number_box:
      number_of_constraints_box = max_number_box
    else:
      number_of_constraints_box = random.choice(range(min_number_box, max_number_box))

    # create tensor for saving
    self.constraints_boxes = torch.zeros(number_of_constraints_box, 2, 3).to(self.device) # boxes, (start-end points), coords

    # create th first boxes
    __, __, box = generate_random_box(self.space_size, 1, reverse_distr=False, max_size_rate=0.1, min_size_rate=0.05)
    self.constraints_boxes[0] = box.to(self.device)

    # create other boxes
    if number_of_constraints_box > 1:
      for i in range(1, number_of_constraints_box):
        __, __, box = generate_random_box(self.space_size, 1, reverse_distr=False, max_size_rate=0.1, min_size_rate=0.05)
        self.constraints_boxes[i] = box.to(self.device)

    # update constraints tensor
    for box in  self.constraints_boxes:
      x1_index, y1_index, z1_index, x2_index, y2_index, z2_index = find_indexes(box, self.space_size, self.state_size)
      self.workspace_constraints[x1_index:x2_index, y1_index:y2_index, z1_index:z2_index] = 1

  def move_face(self, face_dir_indexes, shift_step=1):
    """The functon moves the face by shift_step.
    Inputs:
      face_dir_indexes = [(X or Y or Z), direction of face (e.g. -X or -Y), direction of movement]
      shift_step - movement step
    """
    # extract date
    face_index = face_dir_indexes[0]
    face_pos = face_dir_indexes[1]
    move_dir = face_dir_indexes[2]

    # extract point
    point = self.current_box[face_pos, face_index]

    # move point
    new_point = point + move_dir*shift_step

    #check if point out of space limits
    if new_point > self.space_size*0.5 or new_point < -self.space_size*0.5:
      new_point = new_point*self.space_size*0.5/abs(new_point)


    # if box didn't become the plane return it to the current box tensor
    copy_current_box = self.current_box.clone().detach()
    copy_current_box[face_pos, face_index] = new_point 

    comparison = (copy_current_box[0, :] == copy_current_box[1, :])*1

    if comparison.sum() == 0:
        self.current_box[face_pos, face_index] = new_point

  def update_current_workspace(self):
    """The function transfer geometric date of current box to the tensor type date.
    """
    x1_index, y1_index, z1_index, x2_index, y2_index, z2_index = find_indexes(self.current_box, self.space_size, self.state_size)
    self.workspace_current *= 0
    self.workspace_current[x1_index:x2_index, y1_index:y2_index, z1_index:z2_index] = 1

  def update_workspace(self):
    """
    The function collect date from previous boxes, current box, constraints and task-boxes into
    one observation tensor
    """

    # weights
    w_part_boxes = 0.5
    w_task = 0.25
    w_constraints = 0.8
    self.update_current_workspace()

    # collect part tensors
    # previous boxes=0.5, current_box = 0.75
    part = self.workspace_previous + self.workspace_current
    part = (part >= 1)*1*0.5
    part += self.workspace_current*0.25

    if self.multichannel:
      self.workspace[0, :] = part
      self.workspace[1, :] = self.workspace_task
      self.workspace[2, :] = self.workspace_constraints
    else:
      self.workspace = w_part_boxes*part + w_task*self.workspace_task + w_constraints*self.workspace_constraints

  def calculate_objective(self):
    """The function analyses the design and check how it perfom the task.
    Return:
    rewards - float type.
    """

    #weigths
    touching_weight = 2
    intersection_constraints_weight = -1
    intersection_task_weight = -0.1
    includings_weignhts = -0.2
    first_box_create_weight = 0.5
    first_box_not_creating_weight = -0.1

    # counters
    includings_account_step = 3
    face_touching_counter = 0
    first_box_reward = 0

    ### self.one_task_sucsesfull = False

    # count number of steps without any boxes
    if self.first_step_done == False:
      self.first_box_creating_counter += 1
    else:
      self.first_box_creating_counter = 0

    # reward for the first box
    if self.reward_for_first_box:
      first_box_reward = 1

    #count touching tasks
    local_touched_face_indeces = []
    # self.current_touched_face = []
    face_counter = 0

    for task_face in self.task_faces_clone:
      if check_touch_with_face(task_face, self.current_box):
        local_touched_face_indeces.append(face_counter)

      face_counter += 1

    # get reward only for the new achievements 
    # self.current_touched_face save face that were achieved in the last times
    face_touching_counter = len(local_touched_face_indeces) - len(self.current_touched_face) 
    self.current_touched_face = local_touched_face_indeces

    # check how many steps the current box is inside previous boxes
    if self.previous_boxes != None:
      if check_includings(self.current_box, self.previous_boxes):
        self.includings_counter += 1
      else:
        self.includings_counter = 0

    # check intersections between current box and constraints
    intersections_with_constraints = 0
    if check_intersections(self.current_box, self.constraints_boxes):
      intersections_with_constraints = 1

    # check intersections between current box and task-boxes
    # previously it was used as a reward not as a penalty as now
    intersections_with_task = 0
    if check_intersections(self.current_box, self.task_boxes):
      if self.reward_for_first_task_box_intersection == True: # in order not to get reward for the same task
        intersections_with_task = 1
        self.reward_for_first_task_box_intersection = False

    # sum the rewards
    reward = 0
    reward += face_touching_counter*touching_weight
    reward += intersections_with_constraints*intersection_constraints_weight
    reward += intersections_with_task*intersection_task_weight
    reward += first_box_reward*first_box_create_weight
    reward += self.first_box_creating_counter*first_box_not_creating_weight

    # including penalty works not right away
    if self.includings_counter >= includings_account_step:
      reward += includings_weignhts

    # it was used for the experiments with reward, in order not to get zero reward preaty often
    if reward <= 0:
      reward -= 1
   
    return reward

  def update_previous_workspace(self):
    """The function transfer date from current box to the previous boxes list and update the tensor.
    """
    # if it was the first box
    if self.previous_boxes == None:
      self.previous_boxes = torch.zeros(1, 2, 3).to(self.device) # boxes, (start-end points), coords of points
      self.previous_boxes[0] = self.current_box
      
    else:
      self.previous_boxes = torch.cat((self.previous_boxes, self.current_box[None, :]), 0)
    
    x1_index, y1_index, z1_index, x2_index, y2_index, z2_index = find_indexes(self.current_box, self.space_size, self.state_size)
    self.workspace_previous[x1_index:x2_index, y1_index:y2_index, z1_index:z2_index] = 1

  def render(self, type_rendiring="2D"):
    """The function is used for analyze of working env.
    2D type of rendering will give sleces of workspace tensor.
    3D type will give 3D view in matplotlib
    """

    if type_rendiring == "3D":
      # create a 3D view figure
      fig = plt.figure(figsize=(10, 10))
      ax = Axes3D(fig)

      # define the space limits by creating points on the corner of space
      pos_size = self.space_size/2
      neg_size = -self.space_size/2

      test_cube = torch.tensor([[neg_size, neg_size, neg_size], [pos_size, pos_size, pos_size]])
      verts = poligones_verts(test_cube)
      ax.scatter3D(test_cube[:, 0], test_cube[:, 1], test_cube[:, 2])

      # add faces of current box as a polyfones
      if self.first_step_done:
        verts = poligones_verts(self.current_box)
        ax.add_collection3d(Poly3DCollection(verts, 
            facecolors='forestgreen', linewidths=1, edgecolors='g', alpha=.25))

      # add faces of previous boxes
      if self.previous_boxes != None:
        for box in self.previous_boxes:
          verts = poligones_verts(box)
          ax.add_collection3d(Poly3DCollection(verts, 
              facecolors='forestgreen', linewidths=1, edgecolors='g', alpha=.25))
      
      # add faces of task-boxes
      for box in self.task_boxes:
        verts = poligones_verts(box)
        ax.add_collection3d(Poly3DCollection(verts, 
            facecolors='blue', linewidths=1, edgecolors='b', alpha=.25))
        
      # add faces of constraint-boxes
      for box in self.constraints_boxes:
        verts = poligones_verts(box)
        ax.add_collection3d(Poly3DCollection(verts, 
            facecolors='red', linewidths=1, edgecolors='r', alpha=.25)) 
        

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')

      plt.show()

    if type_rendiring == "2D":

      # give the boxes different weights for diffrent color
      matrix_for_rendering = self.workspace_constraints*0 + self.workspace_constraints*(-1) + (-0.5)*self.workspace_task + self.workspace_previous*1 + 2*self.workspace_current

      # give some limit points maximum and minimum values for color reference
      # matrix_for_rendering[:, 0, 0] = -2
      # matrix_for_rendering[:, 1, 0] = 3
      # matrix_for_rendering = matrix_for_rendering.cpu()

      constraints_mat = self.workspace_constraints.cpu()
      task_mat = self.workspace_task.cpu()
      previous_boxes_mat = self.workspace_previous.cpu()
      current_box_mat = self.workspace_current.cpu()
      constraint_intersaction_mat = ((constraints_mat+previous_boxes_mat+current_box_mat) > 1) * 1

      for i in range(self.state_size):

        plt.figure(figsize=(10,10))
        # plt.pcolor(matrix_for_rendering[i, :, :], edgecolors='k')
        # plt.pcolor(constraints_mat[i, :, :], edgecolors='k', cmap="red")
        # plt.pcolor(task_mat[i, :, :], edgecolors='k', cmap="green")
        plt.spy(constraints_mat[i, :, :], precision=0.1, markersize=16, c="red")
        plt.spy(task_mat[i, :, :], precision=0.1, markersize=16, c="green")
        plt.spy(previous_boxes_mat[i, :, :], precision=0.1, markersize=16, c="gold")
        plt.spy(current_box_mat[i, :, :], precision=0.1, markersize=16, c="yellow")
        plt.spy(constraint_intersaction_mat[i, :, :], precision=0.1, markersize=16, c="purple")

        plt.show()


