import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
import random
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    """任务类型枚举"""
    NETWORK_TRAFFIC = "network_traffic"
    CLOUD_COMPUTING = "cloud_computing"
    SMART_GRID = "smart_grid"
    VEHICLE_ROUTING = "vehicle_routing"

@dataclass
class TaskConfig:
    """任务配置类"""
    task_type: TaskType
    resource_count: int
    demand_pattern: str
    constraint_type: str
    reward_weights: Dict[str, float]
    difficulty_level: float
    
class MetaTaskGenerator:
    """
    元学习任务生成器 - 自动生成多样化的资源分配场景
    
    这是元学习系统的核心组件，能够：
    1. 生成不同领域的资源分配任务
    2. 控制任务难度和多样性
    3. 确保任务间的相关性和差异性
    """
    
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.task_templates = self._initialize_task_templates()
        
    def _initialize_task_templates(self) -> Dict[TaskType, Dict]:
        """初始化任务模板"""
        return {
            TaskType.NETWORK_TRAFFIC: {
                'base_resources': 4,  # 视频、游戏、下载、浏览
                'demand_patterns': ['peak_hours', 'random_burst', 'gradual_increase'],
                'constraints': ['bandwidth_limit', 'latency_sensitive', 'fair_share'],
                'reward_components': ['throughput', 'latency', 'fairness', 'utilization']
            },
            TaskType.CLOUD_COMPUTING: {
                'base_resources': 3,  # CPU、内存、存储
                'demand_patterns': ['workload_spike', 'periodic_batch', 'steady_growth'],
                'constraints': ['cost_budget', 'sla_guarantee', 'energy_limit'],
                'reward_components': ['performance', 'cost', 'sla_compliance', 'energy_efficiency']
            },
            TaskType.SMART_GRID: {
                'base_resources': 5,  # 太阳能、风能、火电、水电、储能
                'demand_patterns': ['daily_cycle', 'weather_dependent', 'industrial_load'],
                'constraints': ['carbon_limit', 'stability_requirement', 'cost_optimization'],
                'reward_components': ['stability', 'carbon_footprint', 'cost', 'renewable_ratio']
            },
            TaskType.VEHICLE_ROUTING: {
                'base_resources': 6,  # 不同类型车辆和路线
                'demand_patterns': ['rush_hour', 'event_driven', 'weather_impact'],
                'constraints': ['fuel_limit', 'time_window', 'capacity_constraint'],
                'reward_components': ['travel_time', 'fuel_consumption', 'customer_satisfaction', 'vehicle_utilization']
            }
        }
    
    def generate_task(self, task_type: TaskType = None, difficulty: float = None) -> TaskConfig:
        """
        生成一个新的任务配置
        
        Args:
            task_type: 指定任务类型，None则随机选择
            difficulty: 任务难度 (0.0-1.0)，None则随机选择
            
        Returns:
            TaskConfig: 生成的任务配置
        """
        if task_type is None:
            task_type = self.rng.choice(list(TaskType))
        
        if difficulty is None:
            difficulty = self.rng.uniform(0.2, 0.9)
        
        template = self.task_templates[task_type]
        
        # 根据难度调整资源数量
        base_count = template['base_resources']
        resource_count = base_count + int(difficulty * 4)  # 最多增加4个资源
        
        # 随机选择需求模式和约束
        demand_pattern = self.rng.choice(template['demand_patterns'])
        constraint_type = self.rng.choice(template['constraints'])
        
        # 生成奖励权重
        reward_weights = self._generate_reward_weights(template['reward_components'], difficulty)
        
        return TaskConfig(
            task_type=task_type,
            resource_count=resource_count,
            demand_pattern=demand_pattern,
            constraint_type=constraint_type,
            reward_weights=reward_weights,
            difficulty_level=difficulty
        )
    
    def _generate_reward_weights(self, components: List[str], difficulty: float) -> Dict[str, float]:
        """生成奖励权重"""
        weights = {}
        
        # 基础权重
        base_weights = self.rng.dirichlet([1.0] * len(components))
        
        # 根据难度调整权重分布
        if difficulty > 0.7:
            # 高难度：权重更不均匀，增加挑战
            concentration = 0.5
        else:
            # 低难度：权重相对均匀
            concentration = 2.0
        
        adjusted_weights = self.rng.dirichlet([concentration] * len(components))
        
        for i, component in enumerate(components):
            weights[component] = float(adjusted_weights[i])
        
        return weights
    
    def generate_task_batch(self, batch_size: int, 
                           task_distribution: Dict[TaskType, float] = None) -> List[TaskConfig]:
        """
        生成一批任务
        
        Args:
            batch_size: 批次大小
            task_distribution: 任务类型分布，None则均匀分布
            
        Returns:
            List[TaskConfig]: 任务配置列表
        """
        if task_distribution is None:
            task_distribution = {task_type: 1.0 for task_type in TaskType}
        
        # 归一化分布
        total_weight = sum(task_distribution.values())
        normalized_dist = {k: v/total_weight for k, v in task_distribution.items()}
        
        tasks = []
        for _ in range(batch_size):
            # 根据分布选择任务类型
            rand_val = self.rng.random()
            cumulative = 0.0
            selected_type = list(TaskType)[0]  # 默认值
            
            for task_type, weight in normalized_dist.items():
                cumulative += weight
                if rand_val <= cumulative:
                    selected_type = task_type
                    break
            
            task = self.generate_task(task_type=selected_type)
            tasks.append(task)
        
        return tasks
    
    def create_curriculum(self, total_tasks: int, 
                         difficulty_progression: str = 'linear') -> List[TaskConfig]:
        """
        创建课程学习序列
        
        Args:
            total_tasks: 总任务数
            difficulty_progression: 难度递增方式 ('linear', 'exponential', 'step')
            
        Returns:
            List[TaskConfig]: 按难度排序的任务序列
        """
        tasks = []
        
        for i in range(total_tasks):
            progress = i / (total_tasks - 1)
            
            if difficulty_progression == 'linear':
                difficulty = 0.1 + 0.8 * progress
            elif difficulty_progression == 'exponential':
                difficulty = 0.1 + 0.8 * (progress ** 2)
            elif difficulty_progression == 'step':
                difficulty = 0.1 + 0.8 * (int(progress * 4) / 4)
            else:
                difficulty = 0.5  # 默认中等难度
            
            task = self.generate_task(difficulty=difficulty)
            tasks.append(task)
        
        return tasks

class MetaEnvironmentWrapper(gym.Env):
    """
    元环境包装器 - 将任务配置转换为可执行的环境
    """
    
    def __init__(self, base_env_class, task_config: TaskConfig):
        super().__init__()
        self.base_env_class = base_env_class
        self.task_config = task_config
        self.base_env = None
        self._setup_environment()
    
    def _setup_environment(self):
        """根据任务配置设置环境"""
        # 创建基础环境
        self.base_env = self.base_env_class()
        
        # 根据任务配置调整环境参数
        self._adapt_observation_space()
        self._adapt_action_space()
        self._adapt_reward_function()
    
    def _adapt_observation_space(self):
        """适配观察空间"""
        # 根据资源数量调整观察空间
        obs_dim = self.task_config.resource_count * 2  # 需求 + 分配
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _adapt_action_space(self):
        """适配动作空间"""
        # 动作数量 = 资源数量 * 2 (增加/减少)
        action_count = self.task_config.resource_count * 2
        self.action_space = spaces.Discrete(action_count)
    
    def _adapt_reward_function(self):
        """适配奖励函数"""
        self.reward_weights = self.task_config.reward_weights
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if self.base_env is None:
            self._setup_environment()
        
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # 根据任务配置调整初始状态
        adapted_obs = self._adapt_observation(obs)
        adapted_info = self._adapt_info(info)
        
        return adapted_obs, adapted_info
    
    def step(self, action):
        """执行动作"""
        # 将元动作转换为基础环境动作
        base_action = self._adapt_action(action)
        
        obs, reward, done, truncated, info = self.base_env.step(base_action)
        
        # 适配输出
        adapted_obs = self._adapt_observation(obs)
        adapted_reward = self._adapt_reward(reward, obs, action)
        adapted_info = self._adapt_info(info)
        
        return adapted_obs, adapted_reward, done, truncated, adapted_info
    
    def _adapt_observation(self, obs):
        """适配观察值"""
        # 简单实现：截断或填充到目标维度
        target_dim = self.observation_space.shape[0]
        if len(obs) > target_dim:
            return obs[:target_dim]
        elif len(obs) < target_dim:
            padded = np.zeros(target_dim)
            padded[:len(obs)] = obs
            return padded
        return obs
    
    def _adapt_action(self, action):
        """适配动作"""
        # 将元动作映射到基础环境动作
        base_action_count = self.base_env.action_space.n
        return action % base_action_count
    
    def _adapt_reward(self, base_reward, obs, action):
        """适配奖励"""
        # 根据任务配置的权重调整奖励
        # 这里是简化实现，实际应该根据具体任务类型计算
        return base_reward * self.task_config.difficulty_level
    
    def _adapt_info(self, info):
        """适配信息"""
        info['task_config'] = self.task_config
        return info

if __name__ == '__main__':
    # 测试元任务生成器
    generator = MetaTaskGenerator(seed=42)
    
    print("🧪 测试元任务生成器")
    
    # 生成单个任务
    task = generator.generate_task()
    print(f"\n📋 生成的任务:")
    print(f"   类型: {task.task_type.value}")
    print(f"   资源数量: {task.resource_count}")
    print(f"   需求模式: {task.demand_pattern}")
    print(f"   约束类型: {task.constraint_type}")
    print(f"   难度: {task.difficulty_level:.2f}")
    print(f"   奖励权重: {task.reward_weights}")
    
    # 生成任务批次
    batch = generator.generate_task_batch(5)
    print(f"\n📦 生成的任务批次 (5个):")
    for i, task in enumerate(batch):
        print(f"   {i+1}. {task.task_type.value} (难度: {task.difficulty_level:.2f})")
    
    # 生成课程学习序列
    curriculum = generator.create_curriculum(10, 'linear')
    print(f"\n📚 课程学习序列 (10个任务):")
    for i, task in enumerate(curriculum):
        print(f"   {i+1}. {task.task_type.value} (难度: {task.difficulty_level:.2f})")
    
    print("\n✅ 元任务生成器测试完成！")
