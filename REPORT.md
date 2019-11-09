[//]: # (Image References)

[vanilla_dqn]: ./vanilla_dqn.png "Vanilla DQN"
[double_dqn]: ./double_dqn.png "Double DQN"
[double_dueling_dqn]: ./double_dueling_dqn.png "Double Dueling DQN"
[dueling_ddqn_per]: ./dueling_ddqn_per.png "Double Dueling DQN with PER"

# Project 1: Navigation

# Project Implementation

## Algorithm

The following variants were implemented:
* vanilla DQN
* double DQN
* dueling DQN
* prioritized experience replay

## Model

The Q Network model is defined in `model.QNetwork`, with 2 fully-connected layers
of 64 neurons each followed by RELU activation. For the output layer, in the
case of vanilla DQN or double DQN, another fully-connected layer of `action_size`
neurons is used. In the case of dueling DQN, the output of the second RELU activation
is split into a value stream of size 1, and a advantage stream of size `action_size`.
The two stream are then combined and with the mean of the advantage stream subtracted
to maintain identifiability.

## Agent

The DQN Agent is defined in `agent.Agent`. The agent handles the learning process
in `Agent.learn()`, where Double DQN and Prioritized Experience Replay update logic
are implemented. 

The Prioritized Experience Replay buffer is extended on top of the `Agent.ReplayBuffer`
class, where a sum tree and a minimum tree are used to efficiently maintain and update
the priorities for sampling.

## Hyperparameters

Following hyperparameters were used:
* replay buffer size: 1e5
* minibatch size: 64 
* discount factor: 0.99 
* target network soft update factor: 1e-3  
* learning rate: 5e-4 
* update every n steps: 4
* starting epsilon: 1.0
* ending epsilon: 0.1
* epsilon decay: 0.995
* starting importance sampling beta: 0.4
* prioritization factor: 0.2

## Results

Vanilla DQN: solved in 549 episodes

![Vanilla DQN][vanilla_dqn]

Double DQN: solved in 594 episodes

![Double DQN][double_dqn]

Dueling Double DQN: solved in 540 episodes (weights saved to `checkpoint.pth`)

![Dueling Double DQN][double_dueling_dqn]

Dueling Double DQN with PER: solved in 649 episodes

![Double Dueling DQN with PER][dueling_ddqn_per]

## Observation

I am suprised to see that Double DQN and PER took longer to solve the environment,
though I may need more hyperparameter tuning for PER to achieve better results.
Perhaps with a more complicated environment the gains would be more obvious.

## Future improvements

It would be a good exercise to implement the rest of Rainbow DQN:
* multi-step returns
* distributional RL
* noisy nets

Also applying the agent to learning from pixels would require updating the model to use
CNNs and stacking frames.

Last but not least more rigorous hyperparamter tuning would provide better results.