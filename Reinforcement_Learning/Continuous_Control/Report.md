# Project report

## Learning algorithm

The learning algorithm used is vanilla DDPG Learning as described in the original paper. As a the vector of state was used instead of an image input, the architecture consists of two four-layer fully connected neural networks, one for the actor and one for the crtic. The actor's role is to output deterministic actions from states, while the critics is to generate Q-values from the states, using targets partially obtained from the actor's outputs. 

It should be noted that as in DQN, a local and target network copy exists for both actor and target, where the target will be updated via a soft-update approach, where parameters are slowly updated iteratively, which has been shown to help convergence.

## Architecture
The actor neural network has following layers:

- Fully connected layer - input: 33 (state size) output: 128
- Batch Normalization Layer
- Fully connected layer - input: 128 output 256
- Fully connected layer - input: 256 output: 4 (action size)
- Relu activation throughout, except tanh activation at end

The critic neural network has following layers:

- Fully connected layer - input: 33 (state size) output: 128
- Batch Normalization Layer
- Fully connected layer - input: 128+4 output 256
- Fully connected layer - input: 256 output: 1
- Relu activation throughout

## Hyperparameters
Final Parameters used in DDPG training:

- Maximum steps per episode: 1000
- Max Number of episodes: 2000
- Training executed every 20 timesteps, 10 repetitions
- Critic learning rate: 1e-3
- Actor learning rate: 1e-3
- TAU value for soft update: 1e-3
- Discount factor Gamma: 0.99
- Minibatch size: 1024
- Buffer size: 1e6

## Results

</p>
<p align="center">
  <img src="https://github.com/EXJUSTICE/Udacity_AI/blob/master/Reinforcement_Learning/Continuous_Control/contcontrol_results2.png?raw=true">
</p>
     
The problem was solved after 100 episodes, where solving is defined as a score of above 30 for over 100 episodes. Note that our early convergence can be attributed to the fact that we had 20 concurrent agents working in our environment, in effect multiplying our effective episodic experiences. It was observed that large buffer sizes (1024) and smaller networks (128,256) seemed to perform better than small buffer sizes and larger networks, which was attributed to the availability of data and improved generalization capability.

Furthermore, it was observed that while 20 DDPG agents could be initialized to match the 20 agents in environment, this was not computationally efficient and produced very poor convergence. Such approaches may be better for an environment where each agent exists in a slightly different conditions, i.e. levels of a game, in order to improve the performance of the agent in a cross-domain manner.
However, for a consistent environment as observed in Reacher, this approach would be redundant, as the same agent can perform across multiple instances.


## Ideas for future work

1. Pixel based inputs
2. Incorporation of different noise functions to encourage exploration
3. Priority experience replay could be implemented to encourage early convergence

