# Project Implementation

## Algorithm

For the project I implemented the following algorithms:
* vanilla DQN
* double DQN
* dueling DQN
* prioritized experience replay

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
Double DQN: solved in 594 episodes
Dueling Double DQN: solved in 540 episodes (weights saved to `checkpoint.pth`)
Dueling Double DQN with PER: solved in 649 episodes

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