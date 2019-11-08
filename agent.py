import numpy as np
import random
from collections import namedtuple

from model import QNetwork
import segment_tree

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

    def __init__(self, state_size, action_size, seed, doubleQ=True, dueling=True, per=False, a=0.2, beta=0.5, e=1e-6):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.doubleQ = doubleQ
        self.dueling = dueling
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, dueling=dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, dueling=dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.per = per
        if self.per:
            # Prioritized Replay memory
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, a)
            self.beta = beta
            self.e = e
        else:
            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.per:
                    experiences = self.memory.sample(self.beta)
                else:
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

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.per:
            states, actions, rewards, next_states, dones, weights, idxs = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        if self.doubleQ:
            selected_action = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, selected_action)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.per:
            priorities = F.l1_loss(Q_expected, Q_targets, reduce=False).detach().cpu().numpy() + self.e
            assert priorities.shape == (BATCH_SIZE, 1)
            for idx, priority in zip(idxs, priorities):
                self.memory.update_priority(idx, priority)

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
        self.buffer_size = buffer_size
        self.memory = []
        self.memory_idx = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.memory_idx] = e
        self.memory_idx += 1
        self.memory_idx = self.memory_idx % self.buffer_size
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        return self._tensors_from_experiences(experiences)
    
    def _tensors_from_experiences(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples with priority."""

    def __init__(self, action_size, buffer_size, batch_size, seed, a):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a (float): sampling parameter
        """
        assert a >= 0 and a <= 1.0
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        
        self.max_priority = 1.0
        self.tree_idx = 0
        self.a = a

        capacity = 1
        while capacity < buffer_size:
            capacity *= 2

        self.sum_tree = segment_tree.SumSegmentTree(capacity)
        self.min_tree = segment_tree.MinSegmentTree(capacity)
    
    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)

        self.sum_tree[self.tree_idx] = self.max_priority ** self.a
        self.min_tree[self.tree_idx] = self.max_priority ** self.a
        self.tree_idx += 1
        self.tree_idx = self.tree_idx % self.buffer_size

    def sample(self, beta=0.5):
        assert len(self.memory) >= self.batch_size
        assert beta >= 0 and beta <= 1.0

        idxs = self._sample_with_priority()
        experiences = [self.memory[idx] for idx in idxs]
        tensors = self._tensors_from_experiences(experiences)
        weights = torch.from_numpy(np.array([self._calc_weight(idx, beta) for idx in idxs])).float().to(device)

        return tensors + (weights, idxs,)
    
    def _sample_with_priority(self):
        idxs = []

        total_priority = self.sum_tree.sum(0, len(self.memory))
        segment_priority = total_priority / self.batch_size

        for i in range(self.batch_size):
            low = segment_priority * i
            high = low + segment_priority
            prefix_sum = random.uniform(low, high)
            idx = self.sum_tree.find_prefixsum_idx(prefix_sum)
            idxs.append(idx)

        return idxs
    
    def _calc_weight(self, idx, beta):
        min_priority = self.min_tree.min(0, len(self.memory)) / self.sum_tree.sum(0, len(self.memory))
        max_weight = (min_priority * len(self.memory)) ** (-1 * beta)
        
        priority = self.sum_tree[idx] / self.sum_tree.sum(0, len(self.memory))
        weight = (priority * len(self.memory)) ** (-1 * beta)
        weight = weight / max_weight
        
        return weight
    
    def update_priority(self, idx, priority):
        assert priority > 0
        assert idx >= 0 and idx < len(self.memory)

        self.sum_tree[idx] = priority ** self.a
        self.min_tree[idx] = priority ** self.a

        self.max_priority = max(self.max_priority, priority)
