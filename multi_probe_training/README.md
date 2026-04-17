# Multi-Probe Training 使用说明

## 文件夹结构

```
multi_probe_training/
├── probe_training_multilayer.py    # 主训练模块
├── test_multi_probe.py             # 测试脚本
├── analyze_multi_probe.py          # 分析脚本
├── run_pipeline.py                 # Python Pipeline
├── run_pipeline.sh                 # Bash Pipeline
├── verify_imports.py               # Import验证脚本
├── MULTI_PROBE_README.md           # 完整文档
├── IMPLEMENTATION_SUMMARY.md       # 实现总结
├── QUICKSTART.md                   # 快速开始指南
├── FILES_CHECKLIST.md              # 文件清单
└── README.md                       # 本文件
```

## 快速开始

### 步骤0：验证环境（推荐）

首次使用前，运行验证脚本确保所有依赖正常：

```bash
cd multi_probe_training
python verify_imports.py
```

如果看到"✓ ALL IMPORTS VERIFIED SUCCESSFULLY!"，说明环境配置正确。

### 方法1：使用Pipeline（推荐）

从multi_probe_training文件夹运行：

```bash
cd multi_probe_training
python run_pipeline.py
```

或使用bash脚本：

```bash
cd multi_probe_training
./run_pipeline.sh
```

### 方法2：手动运行

```bash
cd multi_probe_training

# 1. 测试（可选）
python test_multi_probe.py

# 2. 训练
python probe_training_multilayer.py

# 3. 分析
python analyze_multi_probe.py
```

## 重要说明

### Import路径已修复

所有Python脚本已经配置好从父目录导入模块：
- `config.py`
- `model_utils.py`
- `data_preparation.py`

你不需要修改任何import语句，直接运行即可。

### 输出文件位置

所有输出文件会保存到父目录的相应文件夹：
- 模型文件：`../models/`
- 图片文件：`../figures/`

### 数据集位置

确保数据集在父目录：`../data/pilot_dataset.json`

如果没有，先运行：
```bash
cd ..
python data_preparation.py
cd multi_probe_training
```

## 文档说明

1. **QUICKSTART.md** - 从这里开始，包含详细步骤
2. **MULTI_PROBE_README.md** - 完整技术文档
3. **IMPLEMENTATION_SUMMARY.md** - 实现细节
4. **FILES_CHECKLIST.md** - 文件清单

## 常见问题

### Q: 运行时提示找不到模块？
**A**: 确保你在`multi_probe_training`文件夹内运行脚本，脚本会自动添加父目录到Python路径。

### Q: 输出文件在哪里？
**A**: 
- 模型：`../models/multi_position_probes.pkl`
- 结果：`../models/multi_probe_results.json`
- 图片：`../figures/probe_mse_heatmap.png` 等

### Q: 如何查看结果？
**A**: 
```bash
# 查看热力图
open ../figures/probe_mse_heatmap.png

# 查看分析报告
cat ../models/analysis_summary_report.json
```

## 完整工作流示例

```bash
# 1. 进入文件夹
cd "/Users/eureka/Research/Graduation Thesis/CotAnchor/multi_probe_training"

# 2. 运行pipeline
python run_pipeline.py

# 3. 查看结果
open ../figures/probe_mse_heatmap.png
cat ../models/analysis_summary_report.json
```

## 技术细节

### Import机制

每个Python脚本开头都有：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

这会将父目录（CotAnchor）添加到Python路径，使得可以导入：
- `config`
- `model_utils`
- `data_preparation`

### 相对路径

所有文件路径都使用`config.py`中定义的路径：
- `MODELS_DIR` → `../models/`
- `FIGURES_DIR` → `../figures/`
- `DATA_DIR` → `../data/`

## 下一步

1. 阅读 `QUICKSTART.md` 了解详细使用方法
2. 运行 `python run_pipeline.py` 开始训练
3. 查看生成的可视化结果
4. 阅读 `IMPLEMENTATION_SUMMARY.md` 了解实现细节

祝实验顺利！🚀
