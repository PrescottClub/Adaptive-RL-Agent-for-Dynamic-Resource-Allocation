import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, n_observations, n_actions):
        """
        Initialize the DQN network.
        :param n_observations: int, dimension of the observation space
        :param n_actions: int, number of possible actions
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: torch.Tensor, input tensor (state observations)
        :return: torch.Tensor, Q-values for each action
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

if __name__ == '__main__':
    # Example usage:
    # Assuming observation space is 8-dimensional (as in the env placeholder)
    # Assuming action space has 5 discrete actions (as in the env placeholder)
    n_obs = 8
    n_act = 5
    model = DQN(n_obs, n_act)
    print(model)

    # Create a dummy input tensor (batch_size=1, n_observations)
    dummy_input = torch.randn(1, n_obs)
    output = model(dummy_input)
    print("Dummy input:", dummy_input)
    print("Output Q-values:", output)
    print("Output shape:", output.shape) # Should be (1, n_actions) 