#!/usr/bin/env python3
"""
Test script to verify all components of the Adaptive RL Agent project.
"""

import sys
import os
import numpy as np
import torch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_environment():
    """Test the custom environment."""
    print("Testing DynamicTrafficEnv...")
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        
        env = DynamicTrafficEnv()
        state, info = env.reset()
        
        print(f"  ✓ Environment created successfully")
        print(f"  ✓ State shape: {state.shape}")
        print(f"  ✓ Action space: {env.action_space}")
        print(f"  ✓ Observation space: {env.observation_space}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            print(f"  ✓ Step {i+1}: action={action}, reward={reward:.3f}")
        
        env.close()
        print("  ✓ Environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}\n")
        return False

def test_model():
    """Test the DQN model."""
    print("Testing DQN Model...")
    try:
        from src.models.dqn_model import DQN
        
        state_size = 8
        action_size = 5
        model = DQN(state_size, action_size)
        
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass
        dummy_input = torch.randn(1, state_size)
        output = model(dummy_input)
        
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Model test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}\n")
        return False

def test_replay_buffer():
    """Test the replay buffer."""
    print("Testing ReplayBuffer...")
    try:
        from src.utils.replay_buffer import ReplayBuffer
        
        buffer_size = 1000
        batch_size = 32
        buffer = ReplayBuffer(buffer_size, batch_size, seed=42)
        
        print(f"  ✓ ReplayBuffer created successfully")
        
        # Add some experiences
        for i in range(batch_size * 2):
            state = np.random.rand(8)
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            next_state = np.random.rand(8)
            done = np.random.choice([True, False])
            buffer.add(state, action, reward, next_state, done)
        
        print(f"  ✓ Added {len(buffer)} experiences")
        
        # Test sampling
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample()
            print(f"  ✓ Sampling successful")
            print(f"  ✓ Batch shapes: states{states.shape}, actions{actions.shape}")
        
        print(f"  ✓ ReplayBuffer test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ ReplayBuffer test failed: {e}\n")
        return False

def test_agents():
    """Test the DQN and Double DQN agents."""
    print("Testing Agents...")
    try:
        from src.agents.dqn_agent import DQNAgent
        from src.agents.double_dqn_agent import DoubleDQNAgent
        
        state_size = 8
        action_size = 5
        
        # Test DQN Agent
        dqn_agent = DQNAgent(state_size, action_size, seed=42)
        print(f"  ✓ DQN Agent created successfully")
        
        # Test action selection
        dummy_state = np.random.rand(state_size)
        action = dqn_agent.act(dummy_state)
        print(f"  ✓ DQN action selection: {action}")
        
        # Test Double DQN Agent
        ddqn_agent = DoubleDQNAgent(state_size, action_size, seed=42)
        print(f"  ✓ Double DQN Agent created successfully")
        
        # Test action selection
        action = ddqn_agent.act(dummy_state)
        print(f"  ✓ Double DQN action selection: {action}")
        
        print(f"  ✓ Agents test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Agents test failed: {e}\n")
        return False

def test_integration():
    """Test integration between environment and agent."""
    print("Testing Integration...")
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        from src.agents.dqn_agent import DQNAgent
        
        # Create environment and agent
        env = DynamicTrafficEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size, seed=42)
        
        print(f"  ✓ Environment and agent created")
        
        # Run a short episode
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(10):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Agent learning step
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        print(f"  ✓ Episode completed: {step+1} steps, total reward: {total_reward:.3f}")
        print(f"  ✓ Agent epsilon: {agent.epsilon:.3f}")
        print(f"  ✓ Integration test passed!\n")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}\n")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ADAPTIVE RL AGENT - COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        test_environment,
        test_model,
        test_replay_buffer,
        test_agents,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 60)
    print(f"TESTING SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready for training.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train DQN agent: python main_train.py --agent dqn --episodes 1000")
        print("3. Train Double DQN agent: python main_train.py --agent double_dqn --episodes 1000")
        print("4. Compare agents: python main_evaluate.py --mode compare")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == '__main__':
    main() 