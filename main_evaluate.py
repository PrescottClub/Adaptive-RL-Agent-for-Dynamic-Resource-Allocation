import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os

from src.environments.network_traffic_env import DynamicTrafficEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.double_dqn_agent import DoubleDQNAgent
from src.utils.plotters import plot_comparison, plot_environment_metrics, save_training_data

def evaluate_agent(agent, env, n_episodes=100, max_t=1000, render=False):
    """Evaluate a trained agent.
    
    Args:
        agent: Trained agent
        env: Environment
        n_episodes (int): Number of evaluation episodes
        max_t (int): Maximum timesteps per episode
        render (bool): Whether to render the environment
    
    Returns:
        scores (list): Episode scores
        demands_history (list): History of demands
        allocations_history (list): History of allocations
    """
    scores = []
    demands_history = []
    allocations_history = []
    
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        episode_demands = []
        episode_allocations = []
        
        for t in range(max_t):
            action = agent.act(state, eps=0.0)  # No exploration during evaluation
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record environment metrics
            episode_demands.append(info['current_demands'].copy())
            episode_allocations.append(info['current_allocations'].copy())
            
            state = next_state
            score += reward
            
            if render:
                env.render()
            
            if done:
                break
        
        scores.append(score)
        demands_history.extend(episode_demands)
        allocations_history.extend(episode_allocations)
        
        print(f'\rEpisode {i_episode}\tScore: {score:.2f}\tAverage Score: {np.mean(scores):.2f}', end="")
    
    print(f'\nEvaluation completed! Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    
    return scores, demands_history, allocations_history

def load_agent(agent_type, model_path, state_size, action_size):
    """Load a trained agent from file.
    
    Args:
        agent_type (str): 'dqn' or 'double_dqn'
        model_path (str): Path to the model file
        state_size (int): State space size
        action_size (int): Action space size
    
    Returns:
        agent: Loaded agent
    """
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_size, action_size)
    elif agent_type.lower() == 'double_dqn':
        agent = DoubleDQNAgent(state_size, action_size)
    else:
        raise ValueError("agent_type must be 'dqn' or 'double_dqn'")
    
    agent.load(model_path)
    print(f"Loaded {agent_type.upper()} agent from {model_path}")
    
    return agent

def compare_agents(dqn_model_path, ddqn_model_path, n_episodes=100):
    """Compare DQN and Double DQN agents.
    
    Args:
        dqn_model_path (str): Path to DQN model
        ddqn_model_path (str): Path to Double DQN model
        n_episodes (int): Number of evaluation episodes
    """
    # Create environment
    env = DynamicTrafficEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print("=" * 50)
    print("AGENT COMPARISON")
    print("=" * 50)
    
    # Load agents
    dqn_agent = load_agent('dqn', dqn_model_path, state_size, action_size)
    ddqn_agent = load_agent('double_dqn', ddqn_model_path, state_size, action_size)
    
    # Evaluate DQN
    print("\nEvaluating DQN agent...")
    dqn_scores, dqn_demands, dqn_allocations = evaluate_agent(dqn_agent, env, n_episodes)
    
    # Evaluate Double DQN
    print("\nEvaluating Double DQN agent...")
    ddqn_scores, ddqn_demands, ddqn_allocations = evaluate_agent(ddqn_agent, env, n_episodes)
    
    # Print comparison results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"DQN Average Score: {np.mean(dqn_scores):.2f} ± {np.std(dqn_scores):.2f}")
    print(f"Double DQN Average Score: {np.mean(ddqn_scores):.2f} ± {np.std(ddqn_scores):.2f}")
    print(f"Improvement: {((np.mean(ddqn_scores) - np.mean(dqn_scores)) / np.mean(dqn_scores) * 100):.2f}%")
    
    # Plot comparison
    plot_comparison(dqn_scores, ddqn_scores, window_size=min(20, len(dqn_scores)//5))
    
    # Plot environment metrics for the last episode of each agent
    if len(dqn_demands) > 0 and len(ddqn_demands) > 0:
        # Take last 200 steps for visualization
        steps_to_show = min(200, len(dqn_demands))
        plot_environment_metrics(
            dqn_demands[-steps_to_show:], 
            dqn_allocations[-steps_to_show:],
            save_path='results/dqn_environment_metrics.png'
        )
        plot_environment_metrics(
            ddqn_demands[-steps_to_show:], 
            ddqn_allocations[-steps_to_show:],
            save_path='results/ddqn_environment_metrics.png'
        )
    
    # Save comparison results
    os.makedirs('results', exist_ok=True)
    comparison_data = {
        'episode': list(range(1, len(dqn_scores) + 1)),
        'dqn_score': dqn_scores,
        'ddqn_score': ddqn_scores
    }
    save_training_data(comparison_data, 'results/comparison_results.csv')

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agents')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'compare'],
                       help='Evaluation mode: single agent or comparison (default: single)')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'double_dqn'],
                       help='Agent type for single evaluation (default: dqn)')
    parser.add_argument('--model_path', type=str, default='models/dqn_final.pth',
                       help='Path to model file for single evaluation')
    parser.add_argument('--dqn_model', type=str, default='models/dqn_final.pth',
                       help='Path to DQN model for comparison')
    parser.add_argument('--ddqn_model', type=str, default='models/double_dqn_final.pth',
                       help='Path to Double DQN model for comparison')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single agent evaluation
        print("=" * 50)
        print("SINGLE AGENT EVALUATION")
        print("=" * 50)
        print(f"Agent: {args.agent.upper()}")
        print(f"Model: {args.model_path}")
        print(f"Episodes: {args.episodes}")
        
        # Create environment
        env = DynamicTrafficEnv(render_mode='human' if args.render else None)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Load and evaluate agent
        agent = load_agent(args.agent, args.model_path, state_size, action_size)
        scores, demands_history, allocations_history = evaluate_agent(
            agent, env, args.episodes, render=args.render
        )
        
        # Plot results
        from src.utils.plotters import plot_training_progress
        plot_training_progress(scores, window_size=min(20, len(scores)//5), 
                             title=f'{args.agent.upper()} Evaluation Results')
        
        # Plot environment metrics
        if len(demands_history) > 0:
            steps_to_show = min(200, len(demands_history))
            plot_environment_metrics(
                demands_history[-steps_to_show:], 
                allocations_history[-steps_to_show:]
            )
        
        # Save results
        os.makedirs('results', exist_ok=True)
        evaluation_data = {
            'episode': list(range(1, len(scores) + 1)),
            'score': scores
        }
        save_training_data(evaluation_data, f'results/{args.agent}_evaluation.csv')
        
    elif args.mode == 'compare':
        # Agent comparison
        compare_agents(args.dqn_model, args.ddqn_model, args.episodes)
    
    print("\nEvaluation completed!")

if __name__ == '__main__':
    main() 