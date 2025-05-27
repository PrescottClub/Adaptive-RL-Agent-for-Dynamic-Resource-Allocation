import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os
import time

from src.environments.network_traffic_env import DynamicTrafficEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.double_dqn_agent import DoubleDQNAgent
from src.utils.plotters import plot_training_progress, save_training_data

def train_agent(agent_type='dqn', n_episodes=2000, max_t=1000, eps_start=1.0, 
                eps_end=0.01, eps_decay=0.995, target_score=200.0, 
                save_every=500, model_path='models/', plot_every=100):
    """Train a DQN or Double DQN agent.
    
    Args:
        agent_type (str): 'dqn' or 'double_dqn'
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor for decreasing epsilon
        target_score (float): target average score to solve the environment
        save_every (int): save model every N episodes
        model_path (str): path to save models
        plot_every (int): plot progress every N episodes
    """
    
    # Create environment
    env = DynamicTrafficEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Training {agent_type.upper()} agent...")
    
    # Create agent
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_size, action_size, epsilon=eps_start, 
                        epsilon_min=eps_end, epsilon_decay=eps_decay, seed=42)
    elif agent_type.lower() == 'double_dqn':
        agent = DoubleDQNAgent(state_size, action_size, epsilon=eps_start, 
                              epsilon_min=eps_end, epsilon_decay=eps_decay, seed=42)
    else:
        raise ValueError("agent_type must be 'dqn' or 'double_dqn'")
    
    # Create model directory
    os.makedirs(model_path, exist_ok=True)
    
    # Training metrics
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps_history = []                   # epsilon values
    
    start_time = time.time()
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps_history.append(agent.epsilon) # save epsilon value
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.3f}', end="")
        
        # Print progress every 100 episodes
        if i_episode % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {elapsed_time:.1f}s')
        
        # Plot progress
        if i_episode % plot_every == 0:
            plot_training_progress(scores, title=f'{agent_type.upper()} Training Progress')
        
        # Save model
        if i_episode % save_every == 0:
            model_filename = f'{model_path}/{agent_type}_checkpoint_{i_episode}.pth'
            agent.save(model_filename)
            print(f'\nModel saved: {model_filename}')
        
        # Check if environment is solved
        if np.mean(scores_window) >= target_score:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            model_filename = f'{model_path}/{agent_type}_solved.pth'
            agent.save(model_filename)
            break
    
    # Save final model
    final_model_filename = f'{model_path}/{agent_type}_final.pth'
    agent.save(final_model_filename)
    print(f'\nFinal model saved: {final_model_filename}')
    
    # Save training data
    training_data = {
        'episode': list(range(1, len(scores) + 1)),
        'score': scores,
        'epsilon': eps_history
    }
    data_filename = f'{model_path}/{agent_type}_training_data.csv'
    save_training_data(training_data, data_filename)
    
    # Final plot
    plot_training_progress(scores, title=f'{agent_type.upper()} Final Training Results')
    
    return scores, eps_history

def main():
    parser = argparse.ArgumentParser(description='Train DQN or Double DQN agent')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'double_dqn'],
                       help='Agent type to train (default: dqn)')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of training episodes (default: 2000)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--eps_start', type=float, default=1.0,
                       help='Starting epsilon value (default: 1.0)')
    parser.add_argument('--eps_end', type=float, default=0.01,
                       help='Minimum epsilon value (default: 0.01)')
    parser.add_argument('--eps_decay', type=float, default=0.995,
                       help='Epsilon decay rate (default: 0.995)')
    parser.add_argument('--target_score', type=float, default=200.0,
                       help='Target average score (default: 200.0)')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save model every N episodes (default: 500)')
    parser.add_argument('--model_path', type=str, default='models/',
                       help='Path to save models (default: models/)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ADAPTIVE RL AGENT TRAINING")
    print("=" * 50)
    print(f"Agent Type: {args.agent.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps per Episode: {args.max_steps}")
    print(f"Epsilon: {args.eps_start} -> {args.eps_end} (decay: {args.eps_decay})")
    print(f"Target Score: {args.target_score}")
    print(f"Model Path: {args.model_path}")
    print("=" * 50)
    
    # Train the agent
    scores, eps_history = train_agent(
        agent_type=args.agent,
        n_episodes=args.episodes,
        max_t=args.max_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        target_score=args.target_score,
        save_every=args.save_every,
        model_path=args.model_path
    )
    
    print("\nTraining completed!")
    print(f"Final average score: {np.mean(scores[-100:]):.2f}")
    print(f"Best score: {max(scores):.2f}")

if __name__ == '__main__':
    main() 