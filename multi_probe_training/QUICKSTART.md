# 快速开始指南

## 概述

本指南帮助你快速上手使用多位置多层probe训练系统。

## 前置要求

### 1. 环境依赖
```bash
pip install numpy scikit-learn matplotlib seaborn torch transformers tqdm
```

### 2. 数据准备
确保已经运行过数据准备脚本：
```bash
python data_preparation.py
```

这会在`data/`目录下生成`pilot_dataset.json`。

### 3. 模型配置
检查`config.py`中的配置：
- `MODEL_NAME`: 你使用的模型名称
- `HF_TOKEN`: Hugging Face token（如果需要）
- `DEVICE`: "cuda"或"cpu"

## 使用方法

### 方法1：使用Pipeline脚本（推荐）

**Python版本**（推荐）：
```bash
python run_pipeline.py
```

**Bash版本**：
```bash
./run_pipeline.sh
```

Pipeline会引导你完成：
1. 快速测试（可选）
2. 完整训练
3. 深入分析

### 方法2：手动运行

#### 步骤1：快速测试（可选但推荐）
```bash
python test_multi_probe.py
```

这会：
- 使用10个样本快速验证代码
- 测试5个层（而不是全部28层）
- 生成测试可视化
- 验证保存/加载功能

**预期输出**：
```
TESTING MULTI-POSITION PROBE TRAINING
Loading dataset...
Using 10 samples for testing
...
ALL TESTS PASSED!
```

#### 步骤2：完整训练
```bash
python probe_training_multilayer.py
```

**配置参数**（在脚本中修改）：
```python
NUM_LAYERS = 28  # 模型的层数
ALPHA = 1.0      # Ridge正则化强度
```

**预期时间**：
- 小数据集（~100样本）：5-10分钟
- 中等数据集（~400样本）：20-30分钟
- 取决于：样本数、token数、层数、硬件

**输出文件**：
```
models/multi_position_probes.pkl      # 所有probe模型
models/multi_probe_results.json       # 训练结果
figures/probe_mse_heatmap.png        # MSE热力图
figures/probe_mse_statistics.png     # 统计图
```

#### 步骤3：深入分析
```bash
python analyze_multi_probe.py
```

**生成的分析**：
1. 层级趋势分析
2. Token位置分析
3. 最优probe定位
4. 层-Token交互分析
5. 综合报告（JSON）

**输出文件**：
```
models/analysis_summary_report.json
figures/analysis_layer_trends.png
figures/analysis_token_positions.png
figures/analysis_layer_token_interaction.png
```

## 理解输出

### 1. MSE热力图 (`probe_mse_heatmap.png`)

```
      Layer 0  Layer 1  Layer 2  ...  Layer 27
Token 0   [深]    [浅]    [中]   ...   [浅]
Token 1   [浅]    [深]    [浅]   ...   [中]
Token 2   [中]    [中]    [深]   ...   [深]
...
```

- **横轴**：模型层（0到27）
- **纵轴**：Token位置（0到n_tokens-1）
- **颜色**：MSE值
  - 深色（紫色）= 低MSE = 好的预测
  - 浅色（黄色）= 高MSE = 差的预测

**如何解读**：
- 找深色区域 → 这些(layer, token)组合最能预测数值n
- 垂直条纹 → 某些层在所有位置都表现好
- 水平条纹 → 某些token位置在所有层都表现好
- 深色块 → 特定(layer, token)组合特别好

### 2. 统计图 (`probe_mse_statistics.png`)

**左图：MSE by Layer**
- 显示每层的平均MSE
- 找最低点 → 最佳层

**右图：MSE by Token Position**
- 显示每个位置的平均MSE
- 找最低点 → 最关键的token位置

### 3. 分析报告 (`analysis_summary_report.json`)

```json
{
  "overall_statistics": {
    "mean_mse": 1234.56,
    "min_mse": 123.45,
    ...
  },
  "optimal_probe": {
    "layer": 15,
    "token_position": 8,
    "mse": 123.45
  }
}
```

**关键指标**：
- `optimal_probe`: 最佳的(layer, token)组合
- `best_layer`: 平均表现最好的层
- `best_token_position`: 平均表现最好的位置

## 常见问题

### Q1: 训练时CUDA内存不足
**解决方案**：
```python
# 在probe_training_multilayer.py中减少NUM_LAYERS
NUM_LAYERS = 10  # 从28减少到10
```

或者使用CPU：
```python
# 在config.py中
DEVICE = "cpu"
```

### Q2: Token数量不一致错误
**原因**：不同样本的prompt长度不同

**解决方案**：
确保所有样本使用相同的prompt模板，或在`extract_all_positions_data()`中添加padding。

### Q3: 训练时间太长
**解决方案**：
1. 减少样本数：只使用部分数据集
2. 减少层数：`NUM_LAYERS = 10`
3. 使用更快的模型：`LinearRegression`代替`Ridge`

### Q4: 如何解释结果？
**参考**：
- 查看`MULTI_PROBE_README.md`的"预期结果模式"部分
- 运行`analyze_multi_probe.py`获取详细分析
- 检查`analysis_summary_report.json`

## 下一步

### 1. 探索结果
```bash
# 查看所有生成的图片
open figures/
```

### 2. 自定义分析
修改`analyze_multi_probe.py`添加你自己的分析。

### 3. 使用训练好的probe
```python
from probe_training_multilayer import MultiPositionProbe

# 加载probe
probe_system = MultiPositionProbe(num_layers=28)
probe_system.load("multi_position_probes.pkl")

# 使用最优probe进行预测
best_layer = 15
best_token = 8
prediction = probe_system.predict(hidden_state, best_layer, best_token)
```

### 4. 扩展实验
- 尝试不同的正则化强度（ALPHA）
- 对比不同模型的表征模式
- 分析错误样本的表征

## 获取帮助

### 文档
- `MULTI_PROBE_README.md`: 完整文档
- `IMPLEMENTATION_SUMMARY.md`: 实现总结
- 本文件: 快速开始

### 代码注释
所有函数都有详细的docstring，解释参数和返回值。

### 调试
如果遇到问题：
1. 先运行`test_multi_probe.py`确认基本功能正常
2. 检查错误信息中的文件路径和行号
3. 查看生成的日志输出

## 示例工作流

```bash
# 1. 准备环境
cd /Users/eureka/Research/Graduation\ Thesis/CotAnchor

# 2. 快速测试
python test_multi_probe.py

# 3. 完整训练
python probe_training_multilayer.py

# 4. 查看热力图
open figures/probe_mse_heatmap.png

# 5. 深入分析
python analyze_multi_probe.py

# 6. 查看分析报告
cat models/analysis_summary_report.json
```

## 预期结果示例

训练成功后，你应该看到：

```
MULTI-POSITION PROBE TRAINING COMPLETED
Average Test MSE: 1234.56
Min Test MSE: 123.45
Max Test MSE: 5678.90

Optimal probe location:
  Layer: 15
  Token Position: 8
  MSE: 123.45
```

热力图应该显示清晰的模式，例如：
- 某些层（如中间层）整体颜色较深
- 包含数字的token位置颜色较深
- 存在明显的"最优区域"

祝实验顺利！🚀
