#!/usr/bin/env python3
"""
测试notebook功能的脚本
验证所有模块是否能正常导入和运行
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_basic_imports():
    """测试基础库导入"""
    print("🔧 测试基础库导入...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import torch
        print("✅ 基础库导入成功!")
        return True
    except ImportError as e:
        print(f"❌ 基础库导入失败: {e}")
        return False

def test_traditional_rl():
    """测试传统强化学习模块"""
    print("\n🧠 测试传统强化学习模块...")
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        from src.agents.dqn_agent import DQNAgent
        from src.agents.double_dqn_agent import DoubleDQNAgent
        
        # 创建环境
        env = DynamicTrafficEnv()
        state, info = env.reset()
        
        # 创建智能体
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        
        # 测试交互
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"✅ 传统强化学习模块测试成功!")
        print(f"   状态维度: {state_size}, 动作空间: {action_size}")
        print(f"   测试奖励: {reward:.3f}")
        return True
    except Exception as e:
        print(f"❌ 传统强化学习模块测试失败: {e}")
        return False

def test_meta_learning():
    """测试元学习模块"""
    print("\n⚡ 测试元学习模块...")
    try:
        from src.environments.meta_task_generator import MetaTaskGenerator, TaskType
        from src.agents.meta_dqn_agent import MetaDQNAgent
        from src.utils.meta_trainer import MetaTrainer
        
        # 创建任务生成器
        task_generator = MetaTaskGenerator(seed=42)
        
        # 生成测试任务
        task = task_generator.generate_task(TaskType.NETWORK_TRAFFIC, difficulty=0.5)
        
        # 创建元学习智能体
        meta_agent = MetaDQNAgent(
            state_size=8,
            action_size=10,
            lr=1e-3,
            meta_lr=1e-3,
            adaptation_steps=5,
            seed=42
        )
        
        print(f"✅ 元学习模块测试成功!")
        print(f"   任务类型: {task.task_type.value}")
        print(f"   难度级别: {task.difficulty_level:.2f}")
        print(f"   智能体参数: {sum(p.numel() for p in meta_agent.meta_network.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ 元学习模块测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n📊 测试可视化功能...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 创建简单的测试图表
        import numpy as np
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # matplotlib测试
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title("测试图表")
        plt.close()  # 不显示，只测试功能
        
        # plotly测试
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
        
        print("✅ 可视化功能测试成功!")
        return True
    except Exception as e:
        print(f"❌ 可视化功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试notebook功能...")
    print("=" * 50)
    
    results = {
        "基础库": test_basic_imports(),
        "传统强化学习": test_traditional_rl(),
        "元学习": test_meta_learning(),
        "可视化": test_visualization()
    }
    
    print("\n" + "=" * 50)
    print("📋 测试结果总结:")
    
    all_passed = True
    for module, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {module}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过! notebook应该能正常运行!")
        print("\n📚 使用方法:")
        print("   1. 启动Jupyter: jupyter notebook")
        print("   2. 打开: notebooks/complete_system_demo.ipynb")
        print("   3. 运行所有单元格")
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
        print("\n🔧 建议:")
        print("   1. 检查依赖安装: pip install -e .[notebooks]")
        print("   2. 确保在项目根目录运行")
        print("   3. 检查Python环境配置")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
