# 📚 Notebook 使用指南

## 🎯 核心演示文件

### `complete_system_demo.ipynb` - 完整系统演示

这是项目的**核心展示文件**，包含了完整的元学习驱动的自适应资源分配系统演示。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -e .

# 安装notebook相关依赖
pip install -e ".[notebooks]"

# 或者直接安装所有依赖
pip install jupyter notebook seaborn plotly ipywidgets scikit-learn
```

### 2. 启动Jupyter

```bash
# 在项目根目录启动
jupyter notebook

# 或者直接打开特定文件
jupyter notebook notebooks/complete_system_demo.ipynb
```

### 3. 运行演示

1. 打开 `complete_system_demo.ipynb`
2. 点击 "Kernel" → "Restart & Run All"
3. 等待所有单元格执行完成

## 📋 演示内容

### 🔧 1. 系统初始化
- 导入所有必要的库
- 检查模块状态
- 设置可视化样式

### 🌐 2. 环境测试
- 动态资源分配环境演示
- 状态空间和动作空间分析
- 基础交互测试

### 🧠 3. 传统强化学习演示
- DQN和Double DQN训练
- 学习曲线对比
- 性能统计分析

### ⚡ 4. 元学习系统演示
- 多任务环境生成器
- 课程学习序列
- 元学习DQN智能体

### 📊 5. 性能对比分析
- 少样本学习能力对比
- 收敛速度分析
- 跨域迁移能力评估
- 计算效率对比

### 🌍 6. 跨域迁移演示
- 不同领域任务生成
- 知识迁移矩阵
- 迁移成功率分析

### 🏆 7. 创新成果总结
- 突破性创新点
- 技术优势分析
- 应用前景展望
- 商业价值评估

## 🎨 可视化特性

- **交互式图表**: 使用Plotly创建可交互的图表
- **丰富的统计**: 详细的性能对比和分析
- **美观的样式**: 使用Seaborn优化图表外观
- **实时更新**: 支持参数调整和实时结果更新

## ⚠️ 注意事项

### 依赖要求
- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook
- 所有项目依赖（见requirements.txt）

### 运行环境
- 建议在项目根目录运行
- 确保所有模块能正常导入
- 如果遇到导入错误，检查Python路径设置

### 性能考虑
- 某些训练演示可能需要几分钟时间
- 可以调整训练回合数以加快演示速度
- 建议在有GPU的环境中运行以获得更好性能

## 🔧 故障排除

### 常见问题

1. **模块导入失败**
   ```bash
   # 确保在项目根目录
   cd /path/to/Adaptive-RL-Agent-for-Dynamic-Resource-Allocation
   
   # 重新安装依赖
   pip install -e ".[notebooks]"
   ```

2. **Jupyter无法启动**
   ```bash
   # 安装Jupyter
   pip install jupyter notebook
   
   # 检查安装
   jupyter --version
   ```

3. **可视化图表不显示**
   ```bash
   # 安装可视化库
   pip install matplotlib seaborn plotly
   
   # 启用Jupyter扩展
   jupyter nbextension enable --py widgetsnbextension
   ```

4. **内存不足**
   - 减少训练回合数
   - 关闭其他程序释放内存
   - 重启Jupyter内核

## 📞 获取帮助

如果遇到问题，请：

1. 检查依赖是否正确安装
2. 确保在正确的目录运行
3. 查看错误信息并对照故障排除指南
4. 联系项目维护者：prescottchun@163.com

---

**🎉 享受探索我们的元学习驱动的自适应资源分配系统！**
