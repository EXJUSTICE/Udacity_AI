import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,filename=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            filename: if filename stored weights, then load in weights
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        #Two networks, one for running, and one for fixed target
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        if filename:
            weights = torch.load(filename)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    """
    Master learning method called from the main notebook
    Notice experiences is already sampled from memory
    
    """
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        The sampled batch of experience tuples is already provided for you; 
        you need only use the local and target Q-networks to compute the loss, 
        before taking a step towards minimizing the loss.
        
        
            
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            
        Explanation
        1. Predict action by feeding it observations
        2. Call argmax to select the maximum action
        3. Call env.step to get the rnext obs, rewards and so on
        4. Create a stack frame and append on replay buffer
        
        # After this is done (or every so interval) - invoke learn (this methods)
        1. Obtain entire batch of SARSA from experiences as a batch of 64
        2. Predict action using next obs. Q(S,A) @ T+1
        2. Create a discounted reward as a target.  in TF: y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done)
           (This is for the loss function. Note 1-dones exists to ensure when T+1 = done, Q(S,A) should be 0
        3. Predict the actions using current observations Q(S,A) @ T
        5. Calculate the loss and take its derivative.
        4. copy over to target. This is done in soft_update()
        
        May still be confused abou thow otpimzer and loss are related
        https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/10
        https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        
        Recall that during parameter initialization in pytroch we automatically set requires_Grad= True implictly
        loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. Convenient.
        """
        states, actions, rewards, next_states, dones = experiences
        
        # need to find the action that yields the maximum value at Q(s,a) at T+1
        # Detach to make sure its not being modified, able to serve as fixed target
        
        """
        DDQN vs DQN
        if self.__learning_mode['DQN']:
            Q_target_next = self.__qnetwork_target.forward(next_states).max(1)[0].unsqueeze(1).detach()
        else:
            Q_target_next = self.__qnetwork_target.forward(next_states). \
                gather(1, self.__qnetwork_local.forward(next_states).max(1)[1].unsqueeze(1)).detach()
        
        """
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        #Compute Q targets for current state, 
        Q_target = rewards +(gamma*Q_targets_next* (1-dones))
        
        #https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        #Get expected Q_values from our local model, using current states
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        
        loss= F.mse_loss(Q_expected,Q_target)
        #Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)