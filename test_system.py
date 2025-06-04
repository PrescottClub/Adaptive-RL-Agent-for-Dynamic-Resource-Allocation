#!/usr/bin/env python3
"""
系统测试脚本 - 验证所有组件是否正常工作

运行方式：
python test_system.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_basic_imports():
    """测试基础模块导入"""
    print("🧪 测试基础模块导入...")
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        from src.agents.dqn_agent import DQNAgent
        from src.agents.double_dqn_agent import DoubleDQNAgent
        from src.models.dqn_model import DQN
        from src.utils.replay_buffer import ReplayBuffer
        print("✅ 基础模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 基础模块导入失败: {e}")
        return False

def test_meta_learning_imports():
    """测试元学习模块导入"""
    print("🧪 测试元学习模块导入...")
    try:
        from src.environments.meta_task_generator import MetaTaskGenerator, TaskType
        from src.agents.meta_dqn_agent import MetaDQNAgent
        from src.utils.meta_trainer import MetaTrainer
        print("✅ 元学习模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 元学习模块导入失败: {e}")
        return False

def test_environment():
    """测试环境功能"""
    print("🧪 测试环境功能...")
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        
        env = DynamicTrafficEnv()
        state, info = env.reset()
        
        print(f"   状态空间: {env.observation_space.shape}")
        print(f"   动作空间: {env.action_space.n}")
        print(f"   初始状态形状: {state.shape}")
        
        # 测试环境交互
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"   动作: {action}, 奖励: {reward:.3f}")
        print("✅ 环境功能正常")
        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False

def test_basic_agents():
    """测试基础智能体"""
    print("🧪 测试基础智能体...")
    try:
        from src.environments.network_traffic_env import DynamicTrafficEnv
        from src.agents.dqn_agent import DQNAgent
        from src.agents.double_dqn_agent import DoubleDQNAgent
        
        env = DynamicTrafficEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # 测试DQN智能体
        dqn_agent = DQNAgent(state_size, action_size, seed=42)
        state, _ = env.reset()
        action = dqn_agent.act(state)
        print(f"   DQN智能体动作: {action}")
        
        # 测试Double DQN智能体
        ddqn_agent = DoubleDQNAgent(state_size, action_size, seed=42)
        action = ddqn_agent.act(state)
        print(f"   Double DQN智能体动作: {action}")
        
        print("✅ 基础智能体功能正常")
        return True
    except Exception as e:
        print(f"❌ 基础智能体测试失败: {e}")
        return False

def test_meta_learning():
    """测试元学习功能"""
    print("🧪 测试元学习功能...")
    try:
        from src.environments.meta_task_generator import MetaTaskGenerator, TaskType
        from src.agents.meta_dqn_agent import MetaDQNAgent
        
        # 测试任务生成器
        generator = MetaTaskGenerator(seed=42)
        task = generator.generate_task(TaskType.NETWORK_TRAFFIC, difficulty=0.5)
        print(f"   生成任务: {task.task_type.value}, 难度: {task.difficulty_level:.2f}")
        
        # 测试元学习智能体
        meta_agent = MetaDQNAgent(state_size=8, action_size=10, seed=42)
        meta_agent.set_task(task)
        
        dummy_state = np.random.rand(8)
        action = meta_agent.act(dummy_state)
        print(f"   元学习智能体动作: {action}")
        
        print("✅ 元学习功能正常")
        return True
    except Exception as e:
        print(f"❌ 元学习测试失败: {e}")
        return False

def test_notebooks():
    """测试notebook文件"""
    print("🧪 测试notebook文件...")
    try:
        import json
        
        # 测试主要的notebook文件
        notebooks = [
            "notebooks/experiment_analysis.ipynb",
            "notebooks/meta_learning_demo.ipynb"
        ]
        
        for notebook_path in notebooks:
            if os.path.exists(notebook_path):
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   ✅ {notebook_path}: {len(data['cells'])} 个单元格")
            else:
                print(f"   ⚠️ {notebook_path}: 文件不存在")
        
        print("✅ Notebook文件检查完成")
        return True
    except Exception as e:
        print(f"❌ Notebook测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 元学习驱动的自适应资源分配系统 - 系统测试")
    print("=" * 60)
    
    tests = [
        ("基础模块导入", test_basic_imports),
        ("元学习模块导入", test_meta_learning_imports),
        ("环境功能", test_environment),
        ("基础智能体", test_basic_agents),
        ("元学习功能", test_meta_learning),
        ("Notebook文件", test_notebooks),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常！")
        print("\n🚀 可以开始使用以下功能:")
        print("   • python demo_meta_learning.py - 元学习演示")
        print("   • jupyter notebook notebooks/experiment_analysis.ipynb - 实验分析")
        print("   • jupyter notebook notebooks/meta_learning_demo.ipynb - 详细演示")
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
