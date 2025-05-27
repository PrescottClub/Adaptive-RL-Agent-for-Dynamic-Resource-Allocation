import torch
import torch.nn.functional as F
import numpy as np

from .dqn_agent import DQNAgent

class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent - inherits from DQN Agent but modifies target Q-value calculation."""
    
    def __init__(self, state_size, action_size, lr=5e-4, buffer_size=int(1e5), 
                 batch_size=64, gamma=0.99, tau=1e-3, update_every=4, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, seed=None):
        """Initialize a Double DQN Agent object.
        
        Params are the same as DQN Agent.
        """
        # Initialize the parent DQN Agent
        super(DoubleDQNAgent, self).__init__(
            state_size, action_size, lr, buffer_size, batch_size, 
            gamma, tau, update_every, epsilon, epsilon_min, epsilon_decay, seed
        )
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Uses Double DQN target calculation to reduce overestimation bias.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Double DQN: Use local network to select actions, target network to evaluate
        # Get the best actions for next states from the local network
        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        
        # Get Q values for the selected actions from the target network
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    # Example usage
    state_size = 8  # As defined in our environment
    action_size = 5  # As defined in our environment
    
    agent = DoubleDQNAgent(state_size, action_size, seed=42)
    
    # Test action selection
    dummy_state = np.random.rand(state_size)
    action = agent.act(dummy_state)
    print(f"Selected action: {action}")
    
    # Test learning step
    next_state = np.random.rand(state_size)
    reward = -0.5
    done = False
    agent.step(dummy_state, action, reward, next_state, done)
    print(f"Current epsilon: {agent.epsilon:.4f}")
    print("Double DQN Agent initialized successfully!") 