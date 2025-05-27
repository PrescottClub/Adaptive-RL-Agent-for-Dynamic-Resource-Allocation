# Adaptive RL Agent for Dynamic Resource Allocation

## Overview
Designed and implemented a Deep Reinforcement Learning agent (DQN) to optimize resource allocation in complex, simulated dynamic environments. The agent learns adaptive policies to maximize operational efficiency under fluctuating demands and constraints.

## Key Features
Developed using Python with PyTorch for neural network implementation and OpenAI Gym for environment simulation. Employed NumPy for efficient numerical computation and data handling during training and evaluation.

## Tech Stack
- Python
- PyTorch
- OpenAI Gym
- NumPy
- Pandas (for results analysis) 

## Detailed Development Plan (Learning Project Version)

### 1. Project Initialization & Basic Setup (Partially Completed)
*   Git repository creation and configuration.
*   Initial `README.md`.
*   `.gitignore` file.
*   `requirements.txt` for dependencies.

### 2. Environment Design - "Dynamic Network Traffic Manager"
*   **Concept**: Simulate a small network with diverse services (video, gaming, downloads, browsing). The agent allocates bandwidth to maximize Quality of Service (QoS) and user satisfaction under fluctuating demands.
*   **State Space**:
    *   `current_demand_video`, `current_demand_gaming`, `current_demand_download`, `current_demand_browsing` (e.g., 0-100).
    *   `bandwidth_allocation_video`, `bandwidth_allocation_gaming`, `bandwidth_allocation_download`, `bandwidth_allocation_browsing` (percentage).
    *   Optional: `time_of_day_feature` (e.g., morning, afternoon, evening, night).
*   **Action Space**: Discrete actions to adjust bandwidth for each service (e.g., increase/decrease by a fixed step, maintain). Ensure total allocation <= 100%.
*   **Reward Function**:
    *   Penalties for unmet demand (e.g., video buffering, high gaming latency) and wasted bandwidth.
    *   Rewards for meeting demand, prioritizing critical services, and smooth allocation changes (optional).
*   **Implementation**: Custom OpenAI Gym environment (`DynamicTrafficEnv.py`).

### 3. Agent Implementation - Part 1: Classic DQN (Deep Q-Network)
*   **Neural Network (PyTorch)**: Input layer (state dims), 2-3 hidden Dense layers (ReLU), Output layer (Q-values per action).
*   **Experience Replay Buffer**: Stores `(state, action, reward, next_state, done)` tuples.
*   **Target Network**: For stable learning, periodically updated from the main network.
*   **ε-Greedy Exploration**: Balance exploration/exploitation, with decaying ε.
*   **Training Loop**: Interact, collect experiences, sample mini-batches, compute loss (MSE or Huber), update main network (e.g., Adam optimizer), update target network.

### 4. Agent Implementation - Part 2: Double DQN (Comparative Study)
*   **Core Improvement**: Modifies target Q-value calculation to reduce overestimation:
    *   DQN Target: \(Y_t = R_{t+1} + \gamma \max_a Q_{\text{target}}(S_{t+1}, a; \theta_t^-)\)
    *   Double DQN Target: \(Y_t = R_{t+1} + \gamma Q_{\text{target}}(S_{t+1}, \text{argmax}_a Q(S_{t+1}, a; \theta_t); \theta_t^-)\)
*   **Implementation**: Adapt DQN code, primarily in the loss calculation.
*   **Training**: Similar process as DQN.

### 5. Experiments, Analysis & Visualization
*   **Hyperparameter Tuning**: Learning rate, \(\gamma\), \(\epsilon\) (initial/final/decay), network architecture, batch size, replay buffer size.
*   **Performance Metrics**: Total reward per episode, loss curve, environment-specifics (e.g., unmet demand rate, bandwidth utilization, QoS for key services).
*   **Comparative Analysis**: DQN vs. Double DQN learning curves, final performance, stability. Analyze Q-value overestimation if possible.
*   **Visualization**: `Matplotlib` or `Seaborn` for plots.

### 6. "XAI-Lite" - Simple Decision Insights (Optional)
*   Post-training, feed representative states to trained models.
*   Analyze Q-values to understand *why* an agent chooses a particular action in a given state.

### 7. Code Structure & Documentation
*   `src/`:
    *   `environments/network_traffic_env.py`
    *   `agents/dqn_agent.py`, `agents/double_dqn_agent.py`
    *   `models/dqn_model.py` (PyTorch network)
    *   `utils/replay_buffer.py`, `utils/plotters.py`
*   `notebooks/`: For experiments, analysis, visualization.
*   `main_train.py`: Script to train DQN/Double DQN.
*   `main_evaluate.py`: Script to load and evaluate models.
*   **Documentation**: Clear docstrings. Summarize findings in `README.md`.

### 8. Iteration & Future Extensions
*   Explore advanced DQN variants (Dueling DQN, Prioritized Experience Replay).
*   Introduce more complex environment dynamics.
*   Investigate transfer learning possibilities. 