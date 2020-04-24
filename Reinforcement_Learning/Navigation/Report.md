# Project report

## Learning algorithm
<p align="center">
  <img src= "https://github.com/EXJUSTICE/Udacity_AI/blob/master/Reinforcement_Learning/Navigation/qlearn.png?raw=true">
</p>


The learning algorithm used is a vanilla Deep Q Learning as described in the original paper. As a the vector of state was used instead of an image input, the architecture consists of a three layer fully connected neural network. The deep neural network has following layers:

- Fully connected layer - input: 37 (state size) output: 128
- Fully connected layer - input: 128 output 64
- Fully connected layer - input: 64 output: (action size)

Final Parameters used in DQN training:

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.995

## Results

</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/EXJUSTICE/Udacity_AI/master/Reinforcement_Learning/Navigation/NAVIGATION_DQN_RESULTS.png">
</p>
     

```
Episode 100	Average Score: 0.31
Episode 200	Average Score: 3.12
Episode 300	Average Score: 6.19
Episode 400	Average Score: 10.02
Episode 500	Average Score: 12.64
Episode 510	Average Score: 13.03
```
The use of a higher decaying epsilon rate was found to speed up convergence significantly. This can be attributed to the smaller chance of taking random actions, essentially maximizing exploitation


## Ideas for future work

1. Extensive hyperparameter optimization
2. Double Deep Q Networks
3. Prioritized Experience Replay
4. Dueling Deep Q Networks
5. Pixel-based inputs
