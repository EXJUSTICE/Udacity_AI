# Project report

## Learning algorithm

The learning algorithm used is Multi Agent DDPG Learning as described in the original paper. Hence, the approach is similiar to DDPG, apart from the use of two concurrent agents aiming for self-interest in the environment. As such, most of the methods are shared with vanilla DDPG - any changes have been made to accomodate the presence of two agents, and have been specified in the notebook itself.

As a the vector of state was used instead of an image input, the architecture consists of two four-layer fully connected neural networks, one for the actor and one for the crtic. The actor's role is to output deterministic actions from states, while the critics is to generate Q-values from the states, using targets partially obtained from the actor's outputs. 

It should be noted that as in DQN, a local and target network copy exists for both actor and target for both agents, where the target will be updated via a soft-update approach, where parameters are slowly updated iteratively, which has been shown to help convergence.

## Architecture
The actor neural network has following layers:

- Fully connected layer - input: 24 (state size) output: 256
- Fully connected layer - input: 256 output 128
- Fully connected layer - input: 128 output: 4 (action size)
- Relu activation throughout, except tanh activation at end

The critic neural network has following layers:

- Fully connected layer - input: (24+2)*2 ((state size+actionsize)*num_Agents) output:256
- Fully connected layer - input: 256 output 128
- Fully connected layer - input: 128 output: 1
- Relu activation throughout

## Hyperparameters
Final Parameters used in MADDPG training:

- Max Number of episodes: 5000
- Training executed every timestep, 1 instance
- Critic learning rate: 3e-4
- Actor learning rate: 1e-4
- TAU value for soft update: 1e-3
- Discount factor Gamma: 0.999
- Minibatch size: 512
- Buffer size: 1e5

## Results

</p>
<p align="center">
  <img src=https://github.com/EXJUSTICE/Udacity_AI/raw/master/Reinforcement_Learning/Continuous_Control/contcontrol_results2.png?raw=true">
</p>
     
The problem was solved after 1860 episodes, where solving is defined as a score of above 0.5 for over 100 episodes.  
Note that performance degrades after roughly 2200 episodes, which was attributed to overtraining and trapping in local minima.
It should be noted that the learning rates of both actor and critic should be high enough to avoid local minima during the initial training period (<500 episodes). As the critic is shared between both agents and guides their learning, setting its learning rate higher than the acto's was necessary.

Moreover, it was observed that the batch size of the replay buffer had a direct experience to performance, with a buffer size of 512 identified as optimal. Values below this (128,256) fail to reach the same level of performance at the same timestep, while values above this (1024) exhibit even worse performance. This was attributed to a lack of data and excessive noise, respectively.

## Ideas for future work

1. Pixel based inputs
2. Incorporation of different noise functions to encourage exploration
3. Priority experience replay could be implemented to encourage early convergence
4. Training incorporated every X timesteps, for Y repetitions, to account for the non-stationary environment

