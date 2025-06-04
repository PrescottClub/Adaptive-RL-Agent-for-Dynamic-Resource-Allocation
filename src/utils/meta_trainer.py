import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import torch

try:
    from ..environments.meta_task_generator import MetaTaskGenerator, MetaEnvironmentWrapper, TaskConfig
    from ..environments.network_traffic_env import DynamicTrafficEnv
    from ..agents.meta_dqn_agent import MetaDQNAgent
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.environments.meta_task_generator import MetaTaskGenerator, MetaEnvironmentWrapper, TaskConfig
    from src.environments.network_traffic_env import DynamicTrafficEnv
    from src.agents.meta_dqn_agent import MetaDQNAgent

class MetaTrainer:
    """
    元学习训练器 - 实现MAML算法的完整训练流程
    
    核心功能：
    1. 元训练循环（在多个任务上学习如何快速学习）
    2. 快速适应测试（验证在新任务上的适应能力）
    3. 跨域迁移评估（测试不同领域间的知识迁移）
    4. 性能监控和可视化
    """
    
    def __init__(self, agent: MetaDQNAgent, task_generator: MetaTaskGenerator, 
                 base_env_class=DynamicTrafficEnv, config=None):
        self.agent = agent
        self.task_generator = task_generator
        self.base_env_class = base_env_class
        self.config = config or self._default_config()
        
        # 训练统计
        self.meta_losses = []
        self.adaptation_scores = defaultdict(list)
        self.transfer_scores = defaultdict(list)
        self.task_performance = defaultdict(list)
        
        # 时间统计
        self.training_start_time = None
        self.episode_times = deque(maxlen=100)
        
    def _default_config(self):
        """默认配置"""
        return {
            'meta_episodes': 1000,
            'episodes_per_task': 50,
            'adaptation_episodes': 10,
            'meta_batch_size': 4,
            'support_size': 20,
            'query_size': 10,
            'max_steps_per_episode': 200,
            'eval_frequency': 50,
            'save_frequency': 100,
            'early_stopping_patience': 200,
            'target_adaptation_score': 150.0
        }
    
    def meta_train(self, save_path='models/meta_dqn'):
        """
        元训练主循环
        
        Args:
            save_path: 模型保存路径
        """
        self.training_start_time = time.time()
        print("🚀 开始元学习训练...")
        print(f"📊 配置: {self.config}")
        
        best_adaptation_score = float('-inf')
        patience_counter = 0
        
        for meta_episode in range(1, self.config['meta_episodes'] + 1):
            episode_start_time = time.time()
            
            # 生成元批次任务
            meta_batch_tasks = self.task_generator.generate_task_batch(
                self.config['meta_batch_size']
            )
            
            # 收集元批次数据
            meta_batch_data = []
            for task_config in meta_batch_tasks:
                task_data = self._collect_task_data(task_config)
                meta_batch_data.append(task_data)
            
            # 执行元更新
            meta_loss = self.agent.meta_update(meta_batch_data)
            self.meta_losses.append(meta_loss)
            
            # 记录时间
            episode_time = time.time() - episode_start_time
            self.episode_times.append(episode_time)
            
            # 定期评估
            if meta_episode % self.config['eval_frequency'] == 0:
                adaptation_score = self._evaluate_adaptation()
                transfer_score = self._evaluate_transfer()
                
                print(f"📊 元回合 {meta_episode:4d} | "
                      f"元损失: {meta_loss:.4f} | "
                      f"适应分数: {adaptation_score:.2f} | "
                      f"迁移分数: {transfer_score:.2f} | "
                      f"用时: {episode_time:.2f}s")
                
                # 早停检查
                if adaptation_score > best_adaptation_score:
                    best_adaptation_score = adaptation_score
                    patience_counter = 0
                    # 保存最佳模型
                    self.agent.save(f"{save_path}_best.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"🛑 早停触发，最佳适应分数: {best_adaptation_score:.2f}")
                    break
            
            # 定期保存
            if meta_episode % self.config['save_frequency'] == 0:
                self.agent.save(f"{save_path}_episode_{meta_episode}.pth")
                self._save_training_log(f"{save_path}_log.json")
        
        # 训练完成
        total_time = time.time() - self.training_start_time
        print(f"🎉 元训练完成! 总用时: {total_time/60:.2f} 分钟")
        print(f"🏆 最佳适应分数: {best_adaptation_score:.2f}")
        
        # 保存最终模型
        self.agent.save(f"{save_path}_final.pth")
        self._save_training_log(f"{save_path}_final_log.json")
        
        return self.meta_losses, best_adaptation_score
    
    def _collect_task_data(self, task_config: TaskConfig) -> Dict:
        """
        为单个任务收集支持集和查询集数据
        
        Args:
            task_config: 任务配置
            
        Returns:
            Dict: 包含支持集、查询集和任务配置的字典
        """
        # 创建任务环境
        env = MetaEnvironmentWrapper(self.base_env_class, task_config)
        
        # 设置智能体任务
        self.agent.set_task(task_config)
        
        support_data = []
        query_data = []
        
        # 收集支持集数据
        for _ in range(self.config['support_size']):
            state, _ = env.reset()
            action = self.agent.act(state, eps=0.3)  # 较高的探索率
            next_state, reward, done, truncated, _ = env.step(action)
            support_data.append((state, action, reward, next_state, done or truncated))
        
        # 在支持集上快速适应
        self.agent.fast_adapt(support_data)
        
        # 收集查询集数据
        for _ in range(self.config['query_size']):
            state, _ = env.reset()
            action = self.agent.act(state, eps=0.1)  # 较低的探索率
            next_state, reward, done, truncated, _ = env.step(action)
            query_data.append((state, action, reward, next_state, done or truncated))
        
        return {
            'support': support_data,
            'query': query_data,
            'task_config': task_config
        }
    
    def _evaluate_adaptation(self) -> float:
        """
        评估快速适应能力
        
        Returns:
            float: 适应分数（平均回合奖励）
        """
        # 生成新的测试任务
        test_tasks = self.task_generator.generate_task_batch(5)
        adaptation_scores = []
        
        for task_config in test_tasks:
            env = MetaEnvironmentWrapper(self.base_env_class, task_config)
            self.agent.set_task(task_config)
            
            # 收集少量支持数据
            support_data = []
            for _ in range(10):  # 只用10个样本进行适应
                state, _ = env.reset()
                action = self.agent.act(state, eps=0.2)
                next_state, reward, done, truncated, _ = env.step(action)
                support_data.append((state, action, reward, next_state, done or truncated))
            
            # 快速适应
            self.agent.fast_adapt(support_data)
            
            # 测试适应后的性能
            episode_scores = []
            for _ in range(5):  # 测试5个回合
                state, _ = env.reset()
                total_reward = 0
                
                for step in range(self.config['max_steps_per_episode']):
                    action = self.agent.act(state, eps=0.0)  # 贪心策略
                    next_state, reward, done, truncated, _ = env.step(action)
                    total_reward += reward
                    state = next_state
                    
                    if done or truncated:
                        break
                
                episode_scores.append(total_reward)
            
            task_avg_score = np.mean(episode_scores)
            adaptation_scores.append(task_avg_score)
            self.adaptation_scores[task_config.task_type.value].append(task_avg_score)
        
        overall_adaptation_score = np.mean(adaptation_scores)
        return overall_adaptation_score
    
    def _evaluate_transfer(self) -> float:
        """
        评估跨域迁移能力
        
        Returns:
            float: 迁移分数
        """
        # 测试不同任务类型间的迁移
        try:
            from ..environments.meta_task_generator import TaskType
        except ImportError:
            from src.environments.meta_task_generator import TaskType
        
        task_types = list(TaskType)
        transfer_scores = []
        
        for source_type in task_types:
            for target_type in task_types:
                if source_type == target_type:
                    continue
                
                # 在源任务上训练
                source_task = self.task_generator.generate_task(source_type, difficulty=0.5)
                source_env = MetaEnvironmentWrapper(self.base_env_class, source_task)
                self.agent.set_task(source_task)
                
                # 收集源任务数据并适应
                source_data = []
                for _ in range(20):
                    state, _ = source_env.reset()
                    action = self.agent.act(state, eps=0.2)
                    next_state, reward, done, truncated, _ = source_env.step(action)
                    source_data.append((state, action, reward, next_state, done or truncated))
                
                self.agent.fast_adapt(source_data)
                
                # 在目标任务上测试（零样本迁移）
                target_task = self.task_generator.generate_task(target_type, difficulty=0.5)
                target_env = MetaEnvironmentWrapper(self.base_env_class, target_task)
                self.agent.set_task(target_task)
                
                # 测试迁移性能
                state, _ = target_env.reset()
                total_reward = 0
                
                for step in range(self.config['max_steps_per_episode']):
                    action = self.agent.act(state, eps=0.0)
                    next_state, reward, done, truncated, _ = target_env.step(action)
                    total_reward += reward
                    state = next_state
                    
                    if done or truncated:
                        break
                
                transfer_scores.append(total_reward)
                self.transfer_scores[f"{source_type.value}->{target_type.value}"].append(total_reward)
        
        overall_transfer_score = np.mean(transfer_scores) if transfer_scores else 0.0
        return overall_transfer_score
    
    def _save_training_log(self, filepath):
        """保存训练日志"""
        log_data = {
            'meta_losses': self.meta_losses,
            'adaptation_scores': dict(self.adaptation_scores),
            'transfer_scores': dict(self.transfer_scores),
            'task_performance': dict(self.task_performance),
            'config': self.config,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"💾 训练日志已保存到: {filepath}")
    
    def plot_training_progress(self, save_path=None):
        """绘制训练进度"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 元损失曲线
        if self.meta_losses:
            axes[0, 0].plot(self.meta_losses)
            axes[0, 0].set_title('元学习损失')
            axes[0, 0].set_xlabel('元回合')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].grid(True)
        
        # 适应分数
        if self.adaptation_scores:
            for task_type, scores in self.adaptation_scores.items():
                axes[0, 1].plot(scores, label=task_type, marker='o', markersize=3)
            axes[0, 1].set_title('快速适应性能')
            axes[0, 1].set_xlabel('评估次数')
            axes[0, 1].set_ylabel('适应分数')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 迁移分数
        if self.transfer_scores:
            transfer_means = [np.mean(scores) for scores in self.transfer_scores.values()]
            transfer_labels = list(self.transfer_scores.keys())
            axes[1, 0].bar(range(len(transfer_means)), transfer_means)
            axes[1, 0].set_title('跨域迁移性能')
            axes[1, 0].set_xlabel('迁移方向')
            axes[1, 0].set_ylabel('迁移分数')
            axes[1, 0].set_xticks(range(len(transfer_labels)))
            axes[1, 0].set_xticklabels(transfer_labels, rotation=45, ha='right')
            axes[1, 0].grid(True, axis='y')
        
        # 训练时间分布
        if self.episode_times:
            axes[1, 1].hist(self.episode_times, bins=20, alpha=0.7)
            axes[1, 1].set_title('回合用时分布')
            axes[1, 1].set_xlabel('时间 (秒)')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def demonstrate_few_shot_learning(self, task_config: TaskConfig, 
                                    support_sizes: List[int] = [1, 5, 10, 20]):
        """
        演示少样本学习能力
        
        Args:
            task_config: 测试任务配置
            support_sizes: 不同的支持集大小
        """
        print(f"🎯 演示少样本学习: {task_config.task_type.value}")
        
        env = MetaEnvironmentWrapper(self.base_env_class, task_config)
        results = {}
        
        for support_size in support_sizes:
            print(f"📊 测试支持集大小: {support_size}")
            
            # 重置智能体到元网络状态
            self.agent.task_network.load_state_dict(self.agent.meta_network.state_dict())
            self.agent.set_task(task_config)
            
            # 收集支持数据
            support_data = []
            for _ in range(support_size):
                state, _ = env.reset()
                action = self.agent.act(state, eps=0.3)
                next_state, reward, done, truncated, _ = env.step(action)
                support_data.append((state, action, reward, next_state, done or truncated))
            
            # 快速适应
            adaptation_losses = self.agent.fast_adapt(support_data)
            
            # 测试性能
            test_scores = []
            for _ in range(10):
                state, _ = env.reset()
                total_reward = 0
                
                for step in range(self.config['max_steps_per_episode']):
                    action = self.agent.act(state, eps=0.0)
                    next_state, reward, done, truncated, _ = env.step(action)
                    total_reward += reward
                    state = next_state
                    
                    if done or truncated:
                        break
                
                test_scores.append(total_reward)
            
            avg_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            
            results[support_size] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'adaptation_losses': adaptation_losses
            }
            
            print(f"   平均分数: {avg_score:.2f} ± {std_score:.2f}")
        
        return results

if __name__ == '__main__':
    # 测试元学习训练器
    print("🧪 测试元学习训练器")
    
    # 创建组件
    task_generator = MetaTaskGenerator(seed=42)
    agent = MetaDQNAgent(state_size=8, action_size=5, seed=42)
    trainer = MetaTrainer(agent, task_generator)
    
    # 测试任务数据收集
    test_task = task_generator.generate_task()
    task_data = trainer._collect_task_data(test_task)
    
    print(f"✅ 收集到支持集数据: {len(task_data['support'])} 个")
    print(f"✅ 收集到查询集数据: {len(task_data['query'])} 个")
    
    print("✅ 元学习训练器测试完成！")
