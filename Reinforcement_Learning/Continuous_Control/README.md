# Udacity Reinforcement Learning Project 2: Continuous Control
[//]: # (Image References)
</p>

<p align="center">
  <img src= "https://github.com/EXJUSTICE/Udacity_AI/blob/master/Reinforcement_Learning/Continuous_Control/continuouscontrol.gif?raw=true">
</p>


</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/EXJUSTICE/Udacity_AI/master/Reinforcement_Learning/Navigation/NAVIGATION_DQN_RESULTS.png">
</p>
     

# About

This repository hosts the 2nd project on Udacity Deep Reinforcement Learning nanodegree. The 3D environment contains 20 agents expressed as doubly-joined arms, who can move freely to reach the target locations. A reward of +0.1 is provided for each step where the agent's hand is in the goal area. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible, where the goal area will move. 

The environment is based on the Unity Environment, specifically the Reacher environment, with the environment details provided below.

```
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,  
```
An agent was trained using DDPG Learning to solve this task, which it accomplished (beating the mean score of 30 over 100 episodes) after 100 episodes. Trained with GPU-enabled in the Udacity workspace, although any GPU-enabled instance should work.

# Project Details
Students need to implement DeepRL agent using Python and Pytorch.

The simulation contains a 20 agents that concurrently operate in an environment
At each time step, each has four continuous actions, corresponding to torque applicable to two joints.

The state space has `33` dimensions that cover the agent's position, rotation, velocity, and angular velocities.

The task is episodic, and in order to solve the environment, the agent must
get an average score of +30 over 100 consecutive episodes.

# Getting Started
1. Install Unity ML by consulting the documentation
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

2. Download the Unity ML environment from one of the links below based on your OS:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then unzip the file and place the file in this project folder.

3. Create Conda Environment   

Install conda from conda.io. Create a new Conda environment with Python 3.6.

```bash
conda create --name deeprl python=3.6
source activate deeprl
```

4. Install Dependencies
```bash
cd python
pip install .
```

# Instructions
To train the agent, start jupyter notebook, open `Continuous_Control.ipynb`
and execute! Finished training weights are also provided in the 'checkpoint_actor.pth' and ,checkpoint_critic.pth' files, respectively

# Additional informations
- [Performance Report](./Report.md) : Result report of training score
when using a DDPG agent.
