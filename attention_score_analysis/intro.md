在 Transformer 中，随着新 token 不断生成，**所有 token 的 attention 得分会被重新归一化（softmax）**。当 CoT 变得越来越长时，早期 prompt token（包含你的关键条件）需要与越来越多的新 token"竞争"有限的注意力预算，自然会发生稀释。

这个思路的优势在于：
- **机制直接：** 你直接观察信息"是否被看到"，而非"是否被编码";
- **可解释性强：** Attention 权重本身是概率分布，天然具有可量化性;
- **实时监控：** 可以在每个生成步骤实时追踪，无需训练额外模型。


## 量化指标建议

### 1. **归一化原始注意力权重（Normalized Attention Weight）**
$$A_t = \frac{1}{L} \sum_{l=1}^{L} \text{Attn}^{(l)}(\text{[condition token]}, \text{[current position]})$$

其中 $L$ 是层数。对关键条件 token（如那个数字）在每个 CoT 步骤对当前生成位置的平均注意力。

**稀释度量：** 观察 $A_t$ 随时间步 $t$ 的衰减曲线，如果 $A_t \to 0$，说明该条件被"遗忘"了。
