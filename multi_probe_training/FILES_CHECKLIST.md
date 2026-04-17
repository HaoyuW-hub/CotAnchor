# 项目文件清单

## 已创建的文件

### 核心代码文件

1. **probe_training_multilayer.py** (主训练模块)
   - `MultiPositionProbe`类：管理多位置多层probe
   - `extract_all_positions_data()`：提取所有隐藏状态
   - `visualize_mse_heatmap()`：生成MSE热力图
   - `visualize_mse_statistics()`：生成统计图
   - `save_results_json()`：保存结果到JSON

2. **test_multi_probe.py** (测试脚本)
   - 快速验证代码正确性
   - 使用小数据集（10样本，5层）
   - 测试所有核心功能

3. **analyze_multi_probe.py** (分析脚本)
   - 层级趋势分析
   - Token位置分析
   - 最优probe定位
   - 层-Token交互分析
   - 生成综合报告

4. **run_pipeline.py** (Python Pipeline)
   - 自动化完整流程
   - 交互式引导
   - 错误处理

5. **run_pipeline.sh** (Bash Pipeline)
   - Shell脚本版本
   - 已添加执行权限

### 文档文件

1. **MULTI_PROBE_README.md** (完整文档)
   - 概述和核心特性
   - 使用方法和配置
   - 代码结构说明
   - 与NumProbe的对比
   - 研究问题和预期结果
   - 故障排除指南

2. **IMPLEMENTATION_SUMMARY.md** (实现总结)
   - 实现概述
   - 创建的文件说明
   - 训练流程详解
   - 与NumProbe的对比
   - 核心设计决策
   - 预期研究发现
   - 技术细节
   - 扩展方向

3. **QUICKSTART.md** (快速开始指南)
   - 前置要求
   - 使用方法（详细步骤）
   - 理解输出
   - 常见问题解答
   - 示例工作流

4. **FILES_CHECKLIST.md** (本文件)
   - 所有文件清单
   - 文件用途说明

## 文件结构

```
CotAnchor/
├── probe_training_multilayer.py    # 主训练模块
├── test_multi_probe.py             # 测试脚本
├── analyze_multi_probe.py          # 分析脚本
├── run_pipeline.py                 # Python Pipeline
├── run_pipeline.sh                 # Bash Pipeline (可执行)
├── MULTI_PROBE_README.md           # 完整文档
├── IMPLEMENTATION_SUMMARY.md       # 实现总结
├── QUICKSTART.md                   # 快速开始指南
├── FILES_CHECKLIST.md              # 本文件
├── probe_training.py               # 原始单probe训练（保留）
├── model_utils.py                  # 模型工具（已存在）
├── config.py                       # 配置文件（已存在）
├── data_preparation.py             # 数据准备（已存在）
├── models/                         # 模型输出目录
│   ├── multi_position_probes.pkl           # 训练后生成
│   ├── multi_probe_results.json            # 训练后生成
│   └── analysis_summary_report.json        # 分析后生成
└── figures/                        # 图片输出目录
    ├── probe_mse_heatmap.png               # 训练后生成
    ├── probe_mse_statistics.png            # 训练后生成
    ├── analysis_layer_trends.png           # 分析后生成
    ├── analysis_token_positions.png        # 分析后生成
    └── analysis_layer_token_interaction.png # 分析后生成
```

## 使用流程

### 推荐流程
```bash
# 1. 阅读文档
cat QUICKSTART.md

# 2. 运行Pipeline
python run_pipeline.py

# 3. 查看结果
open figures/probe_mse_heatmap.png
cat models/analysis_summary_report.json
```

### 手动流程
```bash
# 1. 测试
python test_multi_probe.py

# 2. 训练
python probe_training_multilayer.py

# 3. 分析
python analyze_multi_probe.py
```

## 文件依赖关系

```
config.py
    ↓
model_utils.py
    ↓
data_preparation.py
    ↓
probe_training_multilayer.py ←→ test_multi_probe.py
    ↓
analyze_multi_probe.py
```

Pipeline脚本调用顺序：
```
run_pipeline.py
    ├→ test_multi_probe.py (可选)
    ├→ probe_training_multilayer.py
    └→ analyze_multi_probe.py
```

## 核心功能对比

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| probe_training_multilayer.py | 训练所有probe | 数据集 | probe模型 + MSE矩阵 |
| test_multi_probe.py | 快速测试 | 小数据集 | 测试结果 |
| analyze_multi_probe.py | 深入分析 | 训练好的probe | 分析报告 + 图表 |
| run_pipeline.py | 自动化流程 | 无 | 完整结果 |

## 关键类和函数

### MultiPositionProbe类
```python
class MultiPositionProbe:
    __init__(num_layers, alpha)
    train(X_all, y, test_size)
    predict(X, layer, token_pos)
    save(filename)
    load(filename)
```

### 主要函数
```python
extract_all_positions_data(model_wrapper, dataset, num_layers)
visualize_mse_heatmap(mse_matrix, save_path, title)
visualize_mse_statistics(mse_matrix, save_path)
save_results_json(results, mse_matrix, filename)
```

### 分析函数
```python
analyze_layer_trends(mse_matrix, save_path)
analyze_token_positions(mse_matrix, save_path)
find_optimal_probe(mse_matrix)
analyze_layer_token_interaction(mse_matrix, save_path)
generate_summary_report(mse_matrix, output_path)
```

## 配置参数

### 在probe_training_multilayer.py中
```python
NUM_LAYERS = 28    # 模型层数
ALPHA = 1.0        # Ridge正则化强度
```

### 在config.py中
```python
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEVICE = "cuda"
TARGET_LAYER = 16
```

## 输出文件说明

### 模型文件
- **multi_position_probes.pkl**: 包含所有训练好的probe（pickle格式）
- **multi_probe_results.json**: 训练结果，包括MSE矩阵（JSON格式）
- **analysis_summary_report.json**: 分析报告（JSON格式）

### 图片文件
- **probe_mse_heatmap.png**: 主要热力图（layer × token）
- **probe_mse_statistics.png**: 统计图（按层和按位置）
- **analysis_layer_trends.png**: 层级趋势分析
- **analysis_token_positions.png**: Token位置分析
- **analysis_layer_token_interaction.png**: 层-Token交互分析

## 下一步

1. **运行测试**: `python test_multi_probe.py`
2. **阅读文档**: 查看`QUICKSTART.md`
3. **开始训练**: `python run_pipeline.py`
4. **分析结果**: 查看生成的图片和报告

## 注意事项

- 所有Python脚本都有详细的docstring
- 所有函数都有类型提示
- 错误处理已实现
- 进度条显示（使用tqdm）
- 自动创建输出目录

## 版本信息

- 创建日期: 2026-04-17
- 基于: NumProbe训练范式
- 适用于: CotAnchor项目
- Python版本: 3.8+
- 主要依赖: torch, sklearn, matplotlib, numpy

## 联系和支持

如有问题，请参考：
1. `QUICKSTART.md` - 常见问题解答
2. `MULTI_PROBE_README.md` - 详细文档
3. 代码中的docstring和注释
