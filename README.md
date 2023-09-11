# <p align="center">Box Cad environment.</p>
  Box CAD is a simplified CAD environment for agent training.
  When the environment is reseted, env creates a task for the agent:
    1. updates the space (removes all geometry from the previous episode).
    2. creates cubes-tasks in space that the body of the construction must reach
    3. creates cubes-constraints; their construction by the body must be avoided.

  Available operations with the construction body:
  * create a new body cube
  * independently move the surface of the cube in the direction of the normal or against it.

  **The observation** is a pytorch tensor, interpreted as a geometric space.
  two options are available:
  1. self.multichannel = False. Paytorch tensor size (3, state_size, state_size, state_size). There are three three-dimensional spaces. The first space is the space of details, the value of the elements in the tensor: <br />
      0.5 for the previous body,<br />
      0.75 for the current cube. <br />
    The triple space is the task space, the value of the tensor elements:<br />
      1 - if there is a task body,<br />
      0 - if not. <br />
    The third space is the space of constraints, the value of the elements:<br />
      1 - there are constraints,<br />
      0 - there are no constraints. <br />
      <br />
  2. self.mutechannel = True. Paytorch tensor size (state_size, state_size, state_size).
    All bodies are in one space at once, the values of the elements
    are equal to the sum of the elements from the previous case through 
    weighting coefficients.
    <br/>

  **Actions:**
  full list of actions - range(0, 12).<br />
    0 - create a new cube;<br />
    1 - move the wall +X along the normal axis;<br />
    2 - move the wall +X against the normal axis;<br />
    3 - move the wall -X along the normal axis;<br />
    4 - move the wall -X against the normal axis;<br />
    5 - move the wall +Y along the normal axis;<br />
    6 - move the wall +Y against the normal axis;<br />
    7 - move the wall -Y along the normal axis;<br />
    8 - move the wall -Y against the normal axis;<br />
    9 - move the wall +Z along the normal axis;<br />
    10 - move the wall + Z against the normal axis;<br />
    11 - move the wall -Z along the normal axis;<br />
    12 - move the wall -Z against the normal axis;<br /><br />

  **Rewards**. The essence of the reward system is the sum through weighting 
  coefficients of the following components:<br />
    * the number of times the current cube touches cubes-tasks. 
      In this case, task cubes that were touched in previous times are not taken into account. 
      (in the future the touch area will be taken into account). The weight is 2.<br />
    * The number of intersections of constraint cubes by the current cube. The weight is -1.<br />
    * The number of intersections of task cubes by the current cube.
      This is done to prevent the current cube from slipping through the task surface. 
      The weight is -0.1.<br />
    * Reward for the first box. This is done to encourage the agent to create a box when 
      the space is empty, rather than trying to move walls that are not there. 
      The weight is 0.5.<br />
    * Number of steps without the first box. The weight is -0.1.<br />
    * penalty if a new box is created inside a previously created body. 
      Works after the third step in a row when the box is inside. The penalty is -0.2.
