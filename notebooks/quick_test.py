#!/usr/bin/env python3
"""
快速测试notebook功能
验证核心模块是否能正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("🚀 快速测试notebook功能...")
    print("=" * 40)
    
    # 测试基础导入
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ 基础库导入成功")
    except ImportError as e:
        print(f"❌ 基础库导入失败: {e}")
        return False
    
    # 测试传统RL模块
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        from src.agents.dqn_agent import DQNAgent
        
        env = DynamicTrafficEnv()
        state, _ = env.reset()
        agent = DQNAgent(state_size=len(state), action_size=env.action_space.n)
        action = agent.act(state)
        
        print("✅ 传统强化学习模块正常")
    except Exception as e:
        print(f"❌ 传统强化学习模块失败: {e}")
        return False
    
    # 测试元学习模块
    try:
        from src.environments.meta_task_generator import MetaTaskGenerator, TaskType
        from src.agents.meta_dqn_agent import MetaDQNAgent
        
        task_gen = MetaTaskGenerator(seed=42)
        task = task_gen.generate_task(TaskType.NETWORK_TRAFFIC, difficulty=0.5)
        meta_agent = MetaDQNAgent(state_size=8, action_size=10)
        
        print("✅ 元学习模块正常")
    except Exception as e:
        print(f"❌ 元学习模块失败: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 所有核心模块测试通过!")
    print("\n📚 现在可以运行notebook:")
    print("   jupyter notebook notebooks/complete_system_demo.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
