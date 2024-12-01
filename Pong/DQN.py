import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    
    def __init__(self, s_size,  a_size, fc1_units=32, fc2_units=32):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(s_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, a_size)
        
    def forward(self, state):
        x = state
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=torch.float32)
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    