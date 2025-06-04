import random
from collections import namedtuple, deque
import numpy as np
import torch

# Define the structure of an experience
Experience = namedtuple('Experience', 
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=None, device='cpu'):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device to put tensors on
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # Actions are usually discrete integers, ensure they are LongTensors for embedding or gathering
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # Dones are boolean, convert to float (0 or 1) for calculations
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

if __name__ == '__main__':
    # Example Usage
    buffer_size = 10000
    batch_size = 64
    replay_buffer = ReplayBuffer(buffer_size, batch_size, seed=42)

    # Add some dummy experiences
    for i in range(batch_size * 2):
        dummy_state = np.random.rand(8) # Assuming 8 features for state
        dummy_action = random.randint(0, 4) # Assuming 5 discrete actions
        dummy_reward = random.random()
        dummy_next_state = np.random.rand(8)
        dummy_done = random.choice([True, False])
        replay_buffer.add(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    
    print(f"Buffer length: {len(replay_buffer)}")

    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        print("Sampled states shape:", states.shape)       # Expected: (batch_size, state_dim)
        print("Sampled actions shape:", actions.shape)     # Expected: (batch_size, 1) or (batch_size, action_dim) if continuous
        print("Sampled rewards shape:", rewards.shape)     # Expected: (batch_size, 1)
        print("Sampled next_states shape:", next_states.shape) # Expected: (batch_size, state_dim)
        print("Sampled dones shape:", dones.shape)         # Expected: (batch_size, 1)
    else:
        print("Not enough experiences in buffer to sample a batch.") 