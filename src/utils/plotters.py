import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def plot_training_progress(scores: List[float], window_size: int = 100, 
                          title: str = "Training Progress", save_path: Optional[str] = None):
    """Plot training scores over episodes with moving average.
    
    Args:
        scores: List of episode scores
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = np.arange(len(scores))
    ax.plot(episodes, scores, alpha=0.6, color='lightblue', label='Episode Score')
    
    # Calculate moving average
    if len(scores) >= window_size:
        moving_avg = pd.Series(scores).rolling(window=window_size).mean()
        ax.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size} episodes)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_comparison(dqn_scores: List[float], ddqn_scores: List[float], 
                   window_size: int = 100, save_path: Optional[str] = None):
    """Compare DQN and Double DQN performance.
    
    Args:
        dqn_scores: DQN episode scores
        ddqn_scores: Double DQN episode scores
        window_size: Window size for moving average
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot raw scores
    episodes_dqn = np.arange(len(dqn_scores))
    episodes_ddqn = np.arange(len(ddqn_scores))
    
    ax1.plot(episodes_dqn, dqn_scores, alpha=0.6, color='blue', label='DQN')
    ax1.plot(episodes_ddqn, ddqn_scores, alpha=0.6, color='green', label='Double DQN')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Raw Episode Scores Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot moving averages
    if len(dqn_scores) >= window_size:
        dqn_moving_avg = pd.Series(dqn_scores).rolling(window=window_size).mean()
        ax2.plot(episodes_dqn, dqn_moving_avg, color='blue', linewidth=2, label=f'DQN Moving Avg ({window_size})')
    
    if len(ddqn_scores) >= window_size:
        ddqn_moving_avg = pd.Series(ddqn_scores).rolling(window=window_size).mean()
        ax2.plot(episodes_ddqn, ddqn_moving_avg, color='green', linewidth=2, label=f'Double DQN Moving Avg ({window_size})')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.set_title('Moving Average Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_epsilon_decay(epsilon_values: List[float], save_path: Optional[str] = None):
    """Plot epsilon decay over training.
    
    Args:
        epsilon_values: List of epsilon values over episodes
        save_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(len(epsilon_values))
    ax.plot(episodes, epsilon_values, color='orange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Epsilon Decay Over Training')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_loss_curve(loss_values: List[float], save_path: Optional[str] = None):
    """Plot training loss over time.
    
    Args:
        loss_values: List of loss values
        save_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(len(loss_values))
    ax.plot(steps, loss_values, color='red', alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_environment_metrics(demands_history: List[np.ndarray], 
                           allocations_history: List[np.ndarray],
                           service_names: List[str] = None,
                           save_path: Optional[str] = None):
    """Plot environment-specific metrics over time.
    
    Args:
        demands_history: List of demand arrays over time
        allocations_history: List of allocation arrays over time
        service_names: Names of services (default: ['Video', 'Gaming', 'Download', 'Browsing'])
        save_path: Path to save the plot (optional)
    """
    if service_names is None:
        service_names = ['Video', 'Gaming', 'Download', 'Browsing']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps = np.arange(len(demands_history))
    demands_array = np.array(demands_history)
    allocations_array = np.array(allocations_history)
    
    # Plot demands
    for i, service in enumerate(service_names):
        ax1.plot(steps, demands_array[:, i], label=f'{service} Demand', alpha=0.7)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Demand Level')
    ax1.set_title('Service Demands Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot allocations
    for i, service in enumerate(service_names):
        ax2.plot(steps, allocations_array[:, i], label=f'{service} Allocation', alpha=0.7)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Bandwidth Allocation')
    ax2.set_title('Bandwidth Allocations Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_training_data(data: Dict, filepath: str):
    """Save training data to CSV file.
    
    Args:
        data: Dictionary containing training data
        filepath: Path to save the CSV file
    """
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Training data saved to {filepath}")

if __name__ == '__main__':
    # Example usage
    # Generate dummy data for testing
    episodes = 1000
    dummy_scores = np.random.randn(episodes).cumsum() + np.random.randn(episodes) * 0.1
    dummy_epsilon = [max(0.01, 1.0 * (0.995 ** i)) for i in range(episodes)]
    
    # Test plotting functions
    plot_training_progress(dummy_scores, title="Test Training Progress")
    plot_epsilon_decay(dummy_epsilon)
    
    print("Plotting utilities created successfully!") 