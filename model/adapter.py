import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, dropout):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x # Add residual connection