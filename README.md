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
