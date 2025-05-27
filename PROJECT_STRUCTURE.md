# 📁 Project Structure

This document outlines the structure and organization of the Adaptive RL Agent project.

## 🏗️ Directory Structure

```
Adaptive-RL-Agent-for-Dynamic-Resource-Allocation/
├── 📁 src/                          # Source code package
│   ├── 📁 agents/                   # RL agent implementations
│   │   ├── dqn_agent.py            # Standard DQN agent
│   │   ├── double_dqn_agent.py     # Double DQN agent
│   │   └── __init__.py
│   ├── 📁 environments/             # Custom environments
│   │   ├── network_traffic_env.py   # Dynamic traffic environment
│   │   └── __init__.py
│   ├── 📁 models/                   # Neural network models
│   │   ├── dqn_model.py            # DQN network architecture
│   │   └── __init__.py
│   ├── 📁 utils/                    # Utility functions
│   │   ├── replay_buffer.py        # Experience replay buffer
│   │   ├── plotters.py             # Visualization utilities
│   │   └── __init__.py
│   └── __init__.py
├── 📁 models/                       # Trained model files
│   ├── dqn_final.pth               # Final DQN model
│   ├── dqn_checkpoint_*.pth        # Training checkpoints
│   └── dqn_training_data.csv       # Training metrics
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── experiment_analysis.ipynb   # Main analysis notebook
│   └── README.md                   # Notebook documentation
├── 📁 tests/                        # Test suite
│   ├── test_components.py          # Component testing
│   ├── conftest.py                 # Pytest configuration
│   └── __init__.py
├── 📁 data/                         # Data directory
│   └── README.md                   # Data organization guide
├── 📁 docs/                         # Documentation
│   └── README.md                   # Documentation guide
├── 📁 configs/                      # Configuration files
│   └── default_config.yaml        # Default hyperparameters
├── 📄 main_train.py                # Training script
├── 📄 main_evaluate.py             # Evaluation script
├── 📄 Makefile                     # Development automation
├── 📄 pytest.ini                   # Pytest configuration
├── 📄 pyproject.toml               # Modern Python project config
├── 📄 MANIFEST.in                  # Package manifest
├── 📄 requirements.txt             # Dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 LICENSE                      # MIT License
└── 📄 README.md                    # Project documentation
```

## 🎯 Design Principles

### 1. **Modular Architecture**
- Clear separation of concerns
- Each module has a specific responsibility
- Easy to test and maintain

### 2. **Python Best Practices**
- Follows PEP 8 style guidelines
- Proper package structure with `__init__.py` files
- Modern packaging with `pyproject.toml`

### 3. **Research-Friendly**
- Jupyter notebooks for experimentation
- Configuration files for hyperparameter management
- Comprehensive logging and visualization

### 4. **Production-Ready**
- Comprehensive test suite
- Proper dependency management
- CI/CD friendly structure

## 📦 Package Organization

### Core Components (`src/`)
- **agents/**: RL algorithm implementations
- **environments/**: Custom Gymnasium environments
- **models/**: Neural network architectures
- **utils/**: Shared utilities and helpers

### Scripts (Root Level)
- **main_train.py**: Training pipeline with CLI
- **main_evaluate.py**: Evaluation and comparison
- **test_components.py**: Automated testing

### Configuration
- **configs/**: YAML configuration files
- **pyproject.toml**: Modern Python project configuration

## 🔧 Development Workflow

1. **Code Development**: Work in `src/` directory
2. **Experimentation**: Use Jupyter notebooks
3. **Testing**: Run `test_components.py`
4. **Training**: Execute `main_train.py`
5. **Evaluation**: Run `main_evaluate.py`

## 📊 Data Management

- **models/**: Trained models and checkpoints
- **data/**: Raw and processed datasets
- **results/**: Experimental results (auto-generated)
- **logs/**: Training logs (auto-generated)

## 🚀 Installation Methods

### Development Installation
```bash
pip install -e .
```

### Production Installation
```bash
pip install .
```

### With Optional Dependencies
```bash
pip install -e ".[dev,notebooks]"
```

## 🧪 Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Component Tests**: `test_components.py` for quick validation

## 📈 Monitoring and Logging

- Structured logging configuration
- Training metrics tracking
- Visualization utilities
- Model checkpointing

This structure follows Python packaging best practices and provides a solid foundation for both research and production use. 