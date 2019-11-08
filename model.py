import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=64, dueling=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.dueling = dueling

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)

        if self.dueling:
            self.val = nn.Linear(hidden_units, 1)
            self.adv = nn.Linear(hidden_units, action_size)
        else:
            self.fc3 = nn.Linear(hidden_units, action_size)  

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if self.dueling:
            val = self.val(x)
            adv = self.adv(x)
            adv_mean = adv.mean(dim=1, keepdim=True)

            out = val + adv - adv_mean
        else:
            out = self.fc3(x)

        return out
