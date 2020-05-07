# Udacity Reinforcement Learning Project 3: Multi-Agent Tennis
[//]: # (Image References)
</p>

<p align="center">
  <img src= "https://github.com/EXJUSTICE/Udacity_AI/blob/master/Reinforcement_Learning/Collaboration_and_Competition/tennis_maddpg.gif?raw=true">
</p>


</p>
<p align="center">
  <img src="https://github.com/EXJUSTICE/Udacity_AI/blob/master/Reinforcement_Learning/Collaboration_and_Competition/TennisScore.png?raw=true">
</p>
     

# About

This repository hosts the 1st project on Udacity Deep Reinforcement Learning nanodegree. The goal of the agent is to play tennis and try to defeat the opposing agent. The environment is based on the Unity Environment, specifically the TennisBrain environment, with the environment details provided below.

```
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
```
An agent was trained using MADDPG to solve this task, which it accomplished (beating the mean score of 0.5 over 100 episodes) after ca. 1860 episodes. Trained with GPU-enabled in the Udacity workspace, although any GPU-enabled instance should work.

# Project Details


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


Students need to implement DeepRL agent using Python and Pytorch.

The simulation contains two agents that play against one another environment.

The state space has `24` dimensions and contains the position, velocity, and angle of the agent.

The task is episodic, and in order to solve the environment, the agent must
get an average score of +0.5 over 100 consecutive episodes.

# Getting Started
1. Check [this nanodegree's prerequisite](https://github.com/udacity/deep-reinforcement-learning/#dependencies), and follow the instructions.


2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.


3. Place the file in the root directory of GitHub repository and unzip (or decompress) the file.


# Instructions
To train the agent, start jupyter notebook, open `Tennis_Final.ipynb`
and execute! Finished training weights are also provided in the 'checkpoint.pth' files for both actor and critic networks of both agents

# Additional informations
- [Performance Report](./Report.md) : Result report of training score
when using a MADDPG agent.
