import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicTrafficEnv(gym.Env):
    """Custom Environment for Dynamic Network Traffic Management."""
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super(DynamicTrafficEnv, self).__init__()
        self.render_mode = render_mode

        # Define action and observation space
        # They must be gym.spaces objects
        # Example: action_space for 4 services, each with 3 actions (decrease, keep, increase)
        # This is a simplified example, will need refinement based on the plan.
        # Let's say we have 5 discrete actions: 
        # 0: decrease video, 1: increase video, 2: decrease gaming, 3: increase gaming, 4: do nothing
        # This is a placeholder and needs to be more robust as per the plan.
        self.action_space = spaces.Discrete(5) # Placeholder

        # Example: observation space (normalized demand and current allocation for 4 services)
        # [demand_video, demand_gaming, demand_download, demand_browsing,
        #  alloc_video, alloc_gaming, alloc_download, alloc_browsing]
        # All values normalized between 0 and 1.
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32) # Placeholder

        # Initialize state (e.g., random demands, initial allocations)
        self.current_demands = np.random.rand(4) # video, gaming, download, browsing
        self.current_allocations = np.array([0.25, 0.25, 0.25, 0.25]) # Initial even allocation

        self.max_steps = 200 # Max steps per episode
        self.current_step = 0

    def _get_obs(self):
        return np.concatenate((self.current_demands, self.current_allocations)).astype(np.float32)

    def _get_info(self):
        return {
            "current_demands": self.current_demands,
            "current_allocations": self.current_allocations
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Reset to a new random state for demands
        self.current_demands = np.random.rand(4)
        # Reset allocations or keep them as a continuation, for now, reset to even
        self.current_allocations = np.array([0.25, 0.25, 0.25, 0.25]) 
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = 0

        # Placeholder for action effects and reward calculation
        # This needs to be implemented according to the detailed plan
        # Example: 
        if action == 0: # Decrease video BW
            self.current_allocations[0] = max(0, self.current_allocations[0] - 0.05)
        elif action == 1: # Increase video BW
            self.current_allocations[0] = min(1, self.current_allocations[0] + 0.05)
        # ... and so on for other actions and services

        # Ensure allocations sum to 1 (or handle over/under allocation)
        # For simplicity, let's assume for now we manually re-normalize if needed, 
        # or the agent learns to keep it valid. A better way is to design actions to preserve this.
        total_allocation = np.sum(self.current_allocations)
        if total_allocation > 1.0:
            self.current_allocations /= total_allocation # Normalize
        elif total_allocation == 0 and not np.all(self.current_allocations == 0):
             # Avoid division by zero if all are zero, unless that is a valid state.
             # This case needs careful handling based on problem definition.
             pass 

        # Placeholder reward: -1 for each step to encourage finishing quickly (bad for this problem)
        # A more meaningful reward function is crucial.
        # Example: reward based on how well allocations meet demands
        demand_met_penalty = np.sum(np.abs(self.current_demands - self.current_allocations))
        reward = -demand_met_penalty

        # Simulate changing demands for the next step
        self.current_demands = np.random.rand(4) # New random demands

        observation = self._get_obs()
        info = self._get_info()

        if self.current_step >= self.max_steps:
            truncated = True
        
        # terminated can be set if a critical failure occurs, e.g., all services crash
        # For now, we only use truncation by max_steps

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            return self._render_frame_ansi()
        elif self.render_mode == "human":
            # For human mode, we might use pygame or matplotlib later
            # For now, print to console
            self._render_frame()
            
    def _render_frame(self):
        # Simple console print for now
        print(f"Step: {self.current_step}")
        print(f"Demands: Video={self.current_demands[0]:.2f}, Game={self.current_demands[1]:.2f}, Dl={self.current_demands[2]:.2f}, Web={self.current_demands[3]:.2f}")
        print(f"Allocs:  Video={self.current_allocations[0]:.2f}, Game={self.current_allocations[1]:.2f}, Dl={self.current_allocations[2]:.2f}, Web={self.current_allocations[3]:.2f}")
        print(f"Total Allocation: {np.sum(self.current_allocations):.2f}")
        print("---")

    def _render_frame_ansi(self):
        return (
            f"Step: {self.current_step}\n"
            f"Demands: Video={self.current_demands[0]:.2f}, Game={self.current_demands[1]:.2f}, Dl={self.current_demands[2]:.2f}, Web={self.current_demands[3]:.2f}\n"
            f"Allocs:  Video={self.current_allocations[0]:.2f}, Game={self.current_allocations[1]:.2f}, Dl={self.current_allocations[2]:.2f}, Web={self.current_allocations[3]:.2f}\n"
            f"Total Allocation: {np.sum(self.current_allocations):.2f}\n---\n"
        )

    def close(self):
        pass # Add any cleanup code here

if __name__ == '__main__':
    # Example of using the environment
    env = DynamicTrafficEnv(render_mode='human')
    obs, info = env.reset()
    done = False
    total_reward_accumulated = 0
    for _ in range(5):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_accumulated += reward
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward_accumulated}")
            obs, info = env.reset()
            total_reward_accumulated = 0
    env.close() 