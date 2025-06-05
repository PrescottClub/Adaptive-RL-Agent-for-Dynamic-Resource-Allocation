import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque, OrderedDict
from typing import List, Dict, Tuple, Any

try:
    from ..models.dqn_model import DQN
    from ..utils.replay_buffer import ReplayBuffer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.models.dqn_model import DQN
    from src.utils.replay_buffer import ReplayBuffer

class MetaDQN(nn.Module):
    """
    元学习DQN网络 - 支持快速适应的神经网络架构
    """
    
    def __init__(self, n_observations, n_actions, hidden_size=128, meta_lr=1e-3):
        super(MetaDQN, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.meta_lr = meta_lr
        
        # 主网络层
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
        
        # 元学习特定的适应层
        self.adaptation_layer = nn.Linear(hidden_size, hidden_size)
        
        # 任务嵌入层 - 用于编码任务特征
        self.task_embedding = nn.Linear(10, hidden_size)  # 任务特征维度为10，输出与隐藏层相同
        
    def forward(self, x, task_features=None):
        """
        前向传播
        
        Args:
            x: 状态观察
            task_features: 任务特征向量
        """
        # 基础特征提取
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        
        # 如果有任务特征，进行任务特定的适应
        if task_features is not None:
            # 确保任务特征是正确的形状
            if task_features.dim() == 1:
                task_features = task_features.unsqueeze(0)

            # 生成任务嵌入
            task_emb = F.relu(self.task_embedding(task_features))

            # 将任务嵌入扩展到匹配批次大小
            if task_emb.size(0) == 1 and h2.size(0) > 1:
                task_emb = task_emb.expand(h2.size(0), -1)

            # 直接相加（现在维度应该匹配）
            h2_adapted = h2 + task_emb
        else:
            h2_adapted = h2
        
        # 输出Q值
        q_values = self.layer3(h2_adapted)
        return q_values
    
    def get_task_features(self, task_config):
        """
        从任务配置中提取特征向量
        
        Args:
            task_config: 任务配置对象
            
        Returns:
            torch.Tensor: 任务特征向量
        """
        # 简化的任务特征提取
        features = [
            task_config.resource_count / 10.0,  # 归一化资源数量
            task_config.difficulty_level,       # 难度级别
            len(task_config.reward_weights),     # 奖励组件数量
        ]
        
        # 添加任务类型的one-hot编码
        task_type_encoding = [0.0] * 4
        task_types = ['network_traffic', 'cloud_computing', 'smart_grid', 'vehicle_routing']
        if task_config.task_type.value in task_types:
            idx = task_types.index(task_config.task_type.value)
            task_type_encoding[idx] = 1.0
        
        features.extend(task_type_encoding)
        
        # 填充到固定长度
        while len(features) < 10:
            features.append(0.0)
        
        return torch.tensor(features[:10], dtype=torch.float32)

class MetaDQNAgent:
    """
    元学习DQN智能体 - 实现MAML算法的DQN版本
    
    核心特性：
    1. 快速适应新任务（few-shot learning）
    2. 跨任务知识迁移
    3. 元训练和快速微调
    """
    
    def __init__(self, state_size, action_size, lr=5e-4, meta_lr=1e-3, 
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, 
                 tau=1e-3, update_every=4, adaptation_steps=5, seed=None):
        """
        初始化元学习DQN智能体
        
        Args:
            adaptation_steps: 快速适应的梯度步数
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.adaptation_steps = adaptation_steps
        self.seed = seed
        
        # 设备选择
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 元网络（用于元训练）
        self.meta_network = MetaDQN(state_size, action_size, meta_lr=meta_lr).to(self.device)
        self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=meta_lr)
        
        # 当前任务网络（从元网络复制而来）
        self.task_network = MetaDQN(state_size, action_size, meta_lr=meta_lr).to(self.device)
        self.task_optimizer = optim.Adam(self.task_network.parameters(), lr=lr)
        
        # 目标网络
        self.target_network = MetaDQN(state_size, action_size, meta_lr=meta_lr).to(self.device)
        
        # 经验回放
        self.memory = ReplayBuffer(buffer_size, batch_size, seed, self.device)
        
        # 元学习相关
        self.current_task_config = None
        self.task_features = None
        self.meta_batch_size = 4  # 元批次中的任务数量
        
        # 训练统计
        self.t_step = 0
        self.adaptation_history = []
        
    def set_task(self, task_config):
        """
        设置当前任务并进行快速适应
        
        Args:
            task_config: 任务配置
        """
        self.current_task_config = task_config
        self.task_features = self.meta_network.get_task_features(task_config)
        
        # 检查任务维度信息（现在使用统一维度，此检查主要用于调试）
        expected_obs_dim = task_config.resource_count * 2
        expected_action_dim = task_config.resource_count * 2
        
        # 现在所有任务都使用统一的16维状态空间和16维动作空间
        # 实际的任务特定维度通过观察空间适配器处理
        
        # 从元网络复制参数到任务网络
        self.task_network.load_state_dict(self.meta_network.state_dict())
        
        # 重新初始化优化器
        self.task_optimizer = optim.Adam(self.task_network.parameters(), lr=self.lr)
        
        print(f"🎯 设置新任务: {task_config.task_type.value} (难度: {task_config.difficulty_level:.2f})")
    
    def act(self, state, eps=0.0):
        """
        选择动作
        
        Args:
            state: 当前状态
            eps: epsilon值（用于epsilon-greedy策略）
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.task_network.eval()
        with torch.no_grad():
            if self.task_features is not None:
                task_features_batch = self.task_features.unsqueeze(0).to(self.device)
                action_values = self.task_network(state, task_features_batch)
            else:
                action_values = self.task_network(state)
        self.task_network.train()
        
        # Epsilon-greedy动作选择
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))
    
    def step(self, state, action, reward, next_state, done):
        """
        保存经验并学习
        """
        # 保存经验
        self.memory.add(state, action, reward, next_state, done)
        
        # 学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def learn(self, experiences, gamma):
        """
        从经验中学习（任务特定的学习）
        """
        states, actions, rewards, next_states, dones = experiences
        
        # 准备任务特征
        if self.task_features is not None:
            batch_size = states.size(0)
            task_features_batch = self.task_features.unsqueeze(0).expand(batch_size, -1).to(self.device)
        else:
            task_features_batch = None
        
        # 计算当前Q值
        Q_expected = self.task_network(states, task_features_batch).gather(1, actions)
        
        # 计算目标Q值
        Q_targets_next = self.target_network(next_states, task_features_batch).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # 计算损失
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 优化
        self.task_optimizer.zero_grad()
        loss.backward()
        self.task_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.task_network, self.target_network, self.tau)
    
    def fast_adapt(self, support_data: List[Tuple], adaptation_lr=None):
        """
        快速适应新任务（MAML的内循环）
        
        Args:
            support_data: 支持集数据 [(state, action, reward, next_state, done), ...]
            adaptation_lr: 适应学习率
        """
        if adaptation_lr is None:
            adaptation_lr = self.lr
        
        # 创建临时网络用于适应
        adapted_network = copy.deepcopy(self.task_network)
        adapted_optimizer = optim.SGD(adapted_network.parameters(), lr=adaptation_lr)
        
        adaptation_losses = []
        
        # 执行几步梯度下降
        for step in range(self.adaptation_steps):
            total_loss = 0.0
            
            for state, action, reward, next_state, done in support_data:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
                reward_tensor = torch.tensor([[reward]], dtype=torch.float).to(self.device)
                done_tensor = torch.tensor([[done]], dtype=torch.float).to(self.device)
                
                # 准备任务特征
                if self.task_features is not None:
                    task_features_batch = self.task_features.unsqueeze(0).to(self.device)
                else:
                    task_features_batch = None
                
                # 计算Q值和损失
                Q_current = adapted_network(state_tensor, task_features_batch).gather(1, action_tensor)
                Q_next = adapted_network(next_state_tensor, task_features_batch).detach().max(1)[0].unsqueeze(1)
                Q_target = reward_tensor + (self.gamma * Q_next * (1 - done_tensor))
                
                loss = F.mse_loss(Q_current, Q_target)
                total_loss += loss
            
            # 反向传播和优化
            adapted_optimizer.zero_grad()
            total_loss.backward()
            adapted_optimizer.step()
            
            adaptation_losses.append(total_loss.item())
        
        # 更新任务网络
        self.task_network.load_state_dict(adapted_network.state_dict())
        self.adaptation_history.append(adaptation_losses)
        
        print(f"🔄 快速适应完成，最终损失: {adaptation_losses[-1]:.4f}")
        
        return adaptation_losses
    
    def meta_update(self, meta_batch_data: List[Dict]):
        """
        元更新（MAML的外循环）
        
        Args:
            meta_batch_data: 元批次数据，每个元素包含一个任务的支持集和查询集
        """
        meta_loss = 0.0
        
        for task_data in meta_batch_data:
            support_set = task_data['support']
            query_set = task_data['query']
            task_config = task_data['task_config']
            
            # 设置任务
            old_task_config = self.current_task_config
            self.set_task(task_config)
            
            # 在支持集上快速适应
            self.fast_adapt(support_set)
            
            # 在查询集上计算损失
            query_loss = self._compute_query_loss(query_set)
            meta_loss += query_loss
            
            # 恢复原任务配置
            self.current_task_config = old_task_config
        
        # 元优化
        meta_loss = meta_loss / len(meta_batch_data)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        print(f"🌟 元更新完成，元损失: {meta_loss.item():.4f}")
        
        return meta_loss.item()
    
    def _compute_query_loss(self, query_data: List[Tuple]):
        """计算查询集上的损失"""
        total_loss = 0.0
        
        for state, action, reward, next_state, done in query_data:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
            reward_tensor = torch.tensor([[reward]], dtype=torch.float).to(self.device)
            done_tensor = torch.tensor([[done]], dtype=torch.float).to(self.device)
            
            # 准备任务特征
            if self.task_features is not None:
                task_features_batch = self.task_features.unsqueeze(0).to(self.device)
            else:
                task_features_batch = None
            
            # 计算Q值和损失
            Q_current = self.task_network(state_tensor, task_features_batch).gather(1, action_tensor)
            Q_next = self.task_network(next_state_tensor, task_features_batch).detach().max(1)[0].unsqueeze(1)
            Q_target = reward_tensor + (self.gamma * Q_next * (1 - done_tensor))
            
            loss = F.mse_loss(Q_current, Q_target)
            total_loss += loss
        
        return total_loss
    
    def soft_update(self, local_model, target_model, tau):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'meta_network_state_dict': self.meta_network.state_dict(),
            'task_network_state_dict': self.task_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'task_optimizer_state_dict': self.task_optimizer.state_dict(),
            'adaptation_history': self.adaptation_history
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.meta_network.load_state_dict(checkpoint['meta_network_state_dict'])
        self.task_network.load_state_dict(checkpoint['task_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.task_optimizer.load_state_dict(checkpoint['task_optimizer_state_dict'])
        self.adaptation_history = checkpoint.get('adaptation_history', [])

if __name__ == '__main__':
    # 测试元学习DQN智能体
    print("🧪 测试元学习DQN智能体")
    
    agent = MetaDQNAgent(state_size=8, action_size=5, seed=42)
    
    # 测试动作选择
    dummy_state = np.random.rand(8)
    action = agent.act(dummy_state, eps=0.1)
    print(f"🎯 选择的动作: {action}")
    
    # 测试任务设置
    try:
        from ..environments.meta_task_generator import MetaTaskGenerator, TaskType
    except ImportError:
        from src.environments.meta_task_generator import MetaTaskGenerator, TaskType

    generator = MetaTaskGenerator(seed=42)
    task_config = generator.generate_task(TaskType.NETWORK_TRAFFIC)
    agent.set_task(task_config)
    
    print("✅ 元学习DQN智能体测试完成！")
