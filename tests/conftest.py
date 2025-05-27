"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def project_root_path():
    """Fixture providing the project root path."""
    return project_root

@pytest.fixture
def temp_model_path(tmp_path):
    """Fixture providing a temporary path for model files."""
    return tmp_path / "test_model.pth"

@pytest.fixture
def sample_config():
    """Fixture providing sample configuration for testing."""
    return {
        'learning_rate': 0.001,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'target_update': 10,
        'hidden_size': 128
    } 