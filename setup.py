#!/usr/bin/env python3
"""
Setup script for Adaptive RL Agent for Dynamic Resource Allocation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-rl-agent",
    version="1.0.0",
    author="PrescottClub",
    author_email="prescottchun@163.com",
    description="Deep Reinforcement Learning for Dynamic Resource Allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PrescottClub/Adaptive-RL-Agent-for-Dynamic-Resource-Allocation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
            "tensorboard>=2.8.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-rl-train=main_train:main",
            "adaptive-rl-eval=main_evaluate:main",
            "adaptive-rl-test=test_components:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="reinforcement-learning, deep-learning, resource-allocation, dqn, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/PrescottClub/Adaptive-RL-Agent-for-Dynamic-Resource-Allocation/issues",
        "Source": "https://github.com/PrescottClub/Adaptive-RL-Agent-for-Dynamic-Resource-Allocation",
        "Documentation": "https://github.com/PrescottClub/Adaptive-RL-Agent-for-Dynamic-Resource-Allocation#readme",
    },
) 