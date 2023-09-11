import torch
import shutil
import random
from collections import deque
import copy
import torch.nn.functional as F
import numpy as np

n_anneal_steps = 1e5 # Anneal over 1m steps in paper
epsilon = lambda step: np.clip(1 - 0.9 * (step/n_anneal_steps), 0.1, 1) # Anneal over 1m steps in paper, 100k here

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, name):
    f_path = checkpoint_dir + '/' + name + 'checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

class BasicBuffer:

  def __init__(self, max_size):
    self.max_size = max_size
    self.buffer = deque(maxlen = max_size)

  def push(self, state, action, reward, next_state, done):
    experience = (state, action, np.array([reward]), next_state, done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []
    

    batch = random.sample(self.buffer, batch_size)

    for experience in batch:
      state, action, reward, next_state, done = experience
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
    return len(self.buffer)
  
class CNN(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.keepprobab = 1

        self.input_dim = input_dim # = 1
        self.output_dim = output_dim

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.input_dim, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - self.keepprobab))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - self.keepprobab))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - self.keepprobab))

        #78608 if state_size = 128
        #11664 if state_size = 64
        #2000  if state_size = 32
        self.fc1 = torch.nn.Linear(2000, 512, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keepprobab))
        # L5 Final FC 627 inputs -> 12 outputs
        self.fc2 = torch.nn.Linear(512, self.output_dim, bias=True)
        # initialize parameters
        torch.nn.init.xavier_uniform_(self.fc2.weight) 

    def forward(self, y):
        output = self.layer1(y)
        output = self.layer2(output)
        output = self.layer3(output)
         # Flatten them for FC
        output = output.view(output.size(0), -1)
        # print(output.size())
        output = self.fc1(output)
        output = self.fc2(output)
        return output
    
class DQNAgent:

  def __init__(self, env, device, learning_rate = 2.5e-4, gamma = 0.99, tau = 0.005, buffer_size = 10000):
    """Deep Q-learning agent for CAD gym. 

    arguments:
    env   - the CAD gym environment

    keyword arguments:
    buffer_size - number of elements in minibatch. default 10000
    """
    self.env = env
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.tau = tau
    self.reply_buffer = BasicBuffer(max_size = buffer_size)

    self.device = device

    self.conv_output_dim = 13 # accrording to the number of actions
    self.conv_input_dim = 3   # means that space has three chanels (part body, constrainted space, requasted space)

    self.model = CNN(self.conv_input_dim, self.conv_output_dim).to(self.device)
    self.target_model = CNN(self.conv_input_dim, self.conv_output_dim).to(self.device)


    for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
      target_param.data.copy_(param)

    self.optimizer = torch.optim.Adam(self.model.parameters())

  def get_action(self, state, eps=0.2):

    state = state.float().unsqueeze(0)
    rand_eps = np.random.random()
    action = 0

    if (rand_eps < eps):
      action = self.env.action_sample()
      return action, True

    else:
      qvals = self.model.forward(state)
      action = np.argmax(qvals.cpu().detach().numpy())

      return action, False

  def compute_loss(self, batch):

    states, actions, rewards, new_states, dones = batch
  
    # states = torch.FloatTensor(torch.stack(states))
    states = torch.stack(states)
    actions = torch.LongTensor(actions).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    # next_states = torch.FloatTensor(torch.stack(new_states))
    next_states = torch.stack(new_states)
    dones = torch.FloatTensor(dones).to(self.device)

    actions = actions.view(actions.size(0), 1)
    dones = dones.view(actions.size(0), 1)

    curr_Q = self.model.forward(states).gather(1, actions)
    next_Q = self.target_model.forward(next_states)
    max_next_Q = torch.max(next_Q, 1)[0]
    max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
    
    expected_Q = rewards + (1-dones)*self.gamma*max_next_Q

    loss = F.mse_loss(curr_Q, expected_Q.detach())

    return loss

  def update(self, batch_size):

    batch = self.reply_buffer.sample(batch_size)
    loss = self.compute_loss(batch)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss

  def update_target(self):

    self.target_model = copy.deepcopy(self.model)