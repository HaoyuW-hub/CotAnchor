# Multi-Layer Multi-Position Probe Training

## 概述

这个模块实现了参照NumProbe范式的多位置、多层probe训练系统。与原始的单probe训练不同，这个系统为输入prompt中的**每个token位置**和**每个隐藏层**训练独立的probe，用于预测prompt中的数值n。

## 核心特性

### 1. 训练策略
- **多位置训练**：为prompt中的每个token位置训练独立的probe
- **多层训练**：覆盖模型的所有隐藏层
- **统一目标**：所有probe预测同一个目标（prompt中的数值n）
- **Ridge回归**：使用Ridge回归（带L2正则化）作为probe模型

### 2. 数据结构
```python
# 输入数据结构
X_all: Dict[int, np.ndarray]
# {layer_idx: (n_samples, n_tokens, hidden_dim)}

# 输出MSE矩阵
mse_matrix: np.ndarray
# Shape: (n_layers, n_tokens)
```

### 3. 可视化输出

#### 热力图（主要可视化）
- **横坐标**：Layer（层索引）
- **纵坐标**：Token Position（token位置）
- **颜色深浅**：MSE值（越深表示MSE越低，预测越准确）

#### 统计图
- **左图**：每层的平均MSE（对所有token位置求平均）
- **右图**：每个token位置的平均MSE（对所有层求平均）

## 使用方法

### 1. 基本使用

```bash
cd /Users/eureka/Research/Graduation\ Thesis/CotAnchor
python probe_training_multilayer.py
```

### 2. 配置参数

在脚本的`__main__`部分修改：

```python
# 模型层数（根据你的模型调整）
NUM_LAYERS = 28  # DeepSeek-R1-Distill-Qwen-7B有28层

# Ridge正则化强度
ALPHA = 1.0  # 增大alpha增强正则化，减小过拟合
```

### 3. 输出文件

训练完成后会生成以下文件：

```
models/
├── multi_position_probes.pkl          # 所有训练好的probe模型
└── multi_probe_results.json           # 训练结果（MSE矩阵等）

figures/
├── probe_mse_heatmap.png             # MSE热力图
└── probe_mse_statistics.png          # MSE统计图
```

## 代码结构

### MultiPositionProbe类

```python
class MultiPositionProbe:
    def __init__(self, num_layers: int, alpha: float = 1.0)
    def train(self, X_all: Dict, y: np.ndarray, test_size: float = 0.2)
    def predict(self, X: np.ndarray, layer: int, token_pos: int)
    def save(self, filename: str)
    def load(self, filename: str)
```

### 主要函数

1. **extract_all_positions_data()**
   - 从所有token位置和所有层提取隐藏状态
   - 返回：`X_all` (字典) 和 `y` (目标数值)

2. **visualize_mse_heatmap()**
   - 生成MSE热力图
   - 横轴=layer，纵轴=token，颜色=MSE

3. **visualize_mse_statistics()**
   - 生成统计图：按层和按位置的平均MSE

4. **save_results_json()**
   - 保存训练结果到JSON文件

## 与NumProbe的对比

| 特性 | NumProbe | CotAnchor Multi-Probe |
|------|----------|----------------------|
| Token选择 | 特定位置（数字开始/结束） | 所有token位置 |
| 层覆盖 | 所有层 | 所有层 |
| Probe类型 | Linear/Ridge/Lasso/MLP | Ridge |
| 预测目标 | 多个（a, b, predicted, golden） | 单个（prompt中的n） |
| 训练粒度 | 每层多个probe（按目标） | 每层每位置一个probe |

## 研究问题

这个训练范式可以回答：

1. **层级表征**：哪些层更好地编码了数值信息？
2. **位置依赖**：数值信息在prompt的哪些位置最容易解码？
3. **表征演化**：数值表征如何在层间传播和演化？
4. **关键位置**：是否存在"关键token"位置，其表征质量显著优于其他位置？

## 预期结果模式

### 可能的热力图模式

1. **数字位置高亮**：包含数字的token位置MSE较低
2. **层级趋势**：中间层可能表现最好（类似BERT的发现）
3. **末尾效应**：最后几个token可能整合了全局信息
4. **层级分化**：浅层vs深层的表征质量差异

## 注意事项

1. **内存消耗**：需要存储所有位置和层的隐藏状态，内存需求较大
2. **计算时间**：训练 `n_layers × n_tokens` 个probe需要较长时间
3. **模型层数**：确保`NUM_LAYERS`与你的模型实际层数匹配
4. **数据集大小**：建议至少100个样本以获得稳定的MSE估计

## 扩展方向

1. **不同probe类型**：尝试MLP probe（参考NumProbe的prober_mlp.py）
2. **注意力权重分析**：结合attention weights分析哪些位置被关注
3. **逐层预测**：分析预测值如何在层间变化
4. **错误分析**：分析哪些样本在哪些位置预测失败

## 故障排除

### 问题1：CUDA内存不足
```python
# 减少batch处理，逐样本提取
# 或减少NUM_LAYERS
```

### 问题2：Token数量不一致
```python
# 确保所有样本使用相同的prompt模板
# 或使用padding使所有样本token数相同
```

### 问题3：某些层缺失
```python
# 检查模型实际层数
print(len(model.model.layers))  # 对于LLaMA系列
```

## 参考

- NumProbe: [原始实现](../NumProbe/)
- CotAnchor原始probe训练: `probe_training.py`
