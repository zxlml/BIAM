<div align="center">

# BIAM：双层交互式可加模型（Bilevel Interactive Additive Model）

**BIAM 的 PyTorch 实现：在缺失值、噪声标签与类别失衡场景下学习**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-63%20passed-brightgreen)](#-测试)

[English](./README.md) | **简体中文**

</div>

---

## 📑 目录

- [✨ 项目亮点](#-项目亮点)
- [🔧 安装](#-安装)
- [⚡ 快速开始](#-快速开始)
- [🏗️ 模型架构](#️-模型架构)
- [🧪 数据腐蚀场景](#-数据腐蚀场景)
- [🔬 消融变体](#-消融变体)
- [📁 实验结果](#-实验结果)
- [⚙️ 配置说明](#️-配置说明)
- [🧪 测试](#-测试)
- [☑️ 待办清单](#️-待办清单)
- [🪪 许可证](#-许可证)

## ✨ 项目亮点

真实世界的表格数据很少有干净的。**BIAM** 在同一个可解释框架内应对三大最常见的数据腐蚀问题：

1. **缺失值** —— 显式建模缺失指示截距以及缺失 × 特征的*交互项*，让模型学习缺失本身携带的信号（支持 MCAR / MAR / MNAR 三种机制）。
2. **噪声标签** —— 通过**双层优化**在干净验证集上学习逐样本权重 ν(·;θ)，自动降低被腐蚀训练样本的权重。
3. **类别失衡** —— 同一重加权机制动态补偿 1:N 的失衡训练分布，同时测试集保持自然分布。

核心设计要点：

- **可解释的可加结构**：每个特征通过可学习的形状函数贡献输出，形状函数建立在位于数据分位数处的 hinge 基之上。
- **一阶双层优化**（公式 4–6）：可微的*虚拟步*使得上层超梯度的计算足够廉价——无需二阶导数。
- **强外推能力**：分段线性 hinge 形状函数的外推能力远优于分段常数替代方案（见[实验结果](#-实验结果)）。

## 🔧 安装

```bash
# 克隆仓库
git clone https://github.com/zxlml/BIAM.git
cd BIAM

# （可选）创建虚拟环境
conda create -n biam python=3.10 -y
conda activate biam

# 安装依赖
pip install -r requirements.txt
```

> **Windows 提示**：如遇 OpenMP 重复库报错，请先设置环境变量再运行：
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK='TRUE'
> ```

**最低要求**：Python 3.8+、PyTorch 2.0+、NumPy、scikit-learn、pandas、matplotlib。完全支持纯 CPU 运行。

## ⚡ 快速开始

### 命令行

```bash
# 回归任务：30% 标签噪声 + 20% MCAR 缺失
python biam_main.py --task regression --noise_ratio 0.3 --missing_ratio 0.2

# 分类任务：标签翻转 + 类别失衡 + MNAR 缺失
python biam_main.py --task classification \
    --noise_ratio 0.2 --imbalance_ratio 0.15 \
    --missing_ratio 0.2 --missing_mechanism MNAR

# 细粒度控制噪声模型（论文公式：y <- y + N(mu_e, sigma_e)）
python biam_main.py --task regression --noise_mean 1.0 --noise_std 0.5 --seed 42

# 消融实验：关闭双层重加权（BIAM-B 变体）
python biam_main.py --task regression --use_bilevel False
```

### Python API

```python
from utils.biam_config import BIAMConfig
from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer

# 1. 配置
config = BIAMConfig()
config.task = 'regression'
config.noise_ratio, config.missing_ratio = 0.3, 0.2

# 2. 生成腐蚀数据（train / val / test 三个划分，test 保持干净）
generator = BIAMDataGenerator(config)
train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()

# 3. 构建模型 + 加权网络；将 hinge 节点置于训练数据分位数处
biam_model = BIAMModel(config, config.device)
weighting_network = BIAMWeightingNetwork(config, config.device)
biam_model.additive_model.set_knots(torch.tensor(train_data[0], dtype=torch.float32))

# 4. 双层优化
optimizer = BIAMOptimizer(config, biam_model, weighting_network)
for epoch in range(config.epochs):
    optimizer.train_epoch(train_loader, val_loader, epoch)
    if epoch % 20 == 0:
        print(optimizer.evaluate(test_data))   # {'mse': ..., 'rmse': ..., 'mae': ...}
```

完整的端到端演示（训练、评估、可解释性分析、可视化）：

```bash
python demo_biam.py
```

## 🏗️ 模型架构

### 核心模块总览

```
BIAM/
├── biam_main.py                  # 主入口（命令行）
├── demo_biam.py                  # 端到端演示
├── run_tests.py                  # 测试套件运行器
├── data/                         # 数据处理
│   ├── biam_data_generator.py    # 合成数据 + 腐蚀管线（噪声/失衡/缺失）
│   ├── biam_binarizer.py         # 缺失值指示与特征分箱
│   ├── biam_dataset_loader.py    # 真实数据集加载工具
│   └── biam_data_utils.py        # 数据辅助函数
├── models/                       # 模型组件
│   ├── biam_additive_model.py    # 可加模型：hinge 基、缺失截距与缺失交互（公式 1）
│   ├── biam_weighting_network.py # 逐样本加权网络 ν(·;θ)
│   └── biam_model.py             # 顶层模型封装
├── gradients/                    # 优化算法
│   ├── biam_optimizer.py         # 一阶双层优化器（公式 4–6）
│   ├── biam_bilevel_optimizer.py # 双层优化替代实现
│   └── biam_gradient_methods.py  # 梯度工具
├── utils/                        # 配置、日志与性能工具
│   ├── biam_config.py            # 集中配置 + 校验
│   └── biam_logger.py            # 结构化日志
├── visualization/                # 形状函数、特征重要性、训练曲线
└── tests/                        # 单元 / 功能 / 仿真 / 性能测试
```

### 可加模型（公式 1）

每个特征通过形状函数贡献输出；缺失值既以截距形式进入，也以与其他特征形状函数交互的形式进入：

$$
f(x, m; w) = \beta_0 + \sum_{j} f_j(x_j) + \sum_{j} \beta^{\text{miss}}_j \, m_j + \sum_{j,k,\tau} \alpha_{j,k,\tau} \, \mathbb{I}(m_j = 1) \, h(x_k; \eta_\tau)
$$

其中 $h(x;\eta) = \max(0, x - \eta)$ 为 hinge 基，其节点 $\eta_\tau$ 置于训练数据分位数处；$m_j \in \{0,1\}$ 表示特征 $j$ 是否缺失。

### 一阶双层优化（公式 4–6）

| 步骤 | 更新 | 目的 |
|------|--------|---------|
| **Step 1** | $w^{(t)} = w^{(t-1)} - \gamma_w \nabla_w R\big(\theta^{(t)}, w^{(t-1)}\big)$ | 以 ν 加权损失对模型参数做真实下层更新 |
| **Step 2** | $\hat{w}(\theta) = w^{(t-1)} - \gamma_w \nabla_w R\big(\theta^{(t)}, w^{(t)}\big)$ | 可微的*虚拟步*（保留在自动微分图中） |
| **Step 3** | $\theta^{(t+1)} = \theta^{(t)} - \gamma_\theta \nabla_\theta L_{\text{val}}\big(\hat{w}(\theta)\big)$ | 超梯度经虚拟步回传，完成上层更新 |

加权网络 ν 以**批内标准化的逐样本损失**为输入并输出样本权重；权重按均值归一化，使重加权保持平均梯度尺度不变。

## 🧪 数据腐蚀场景

遵循论文（§4.1）的仿真协议，`BIAMDataGenerator` 复现全部三类腐蚀——**除特别说明外仅作用于训练集**，测试集保持干净：

| 场景 | 机制 | 作用范围 |
|----------|-----------|------------|
| **标签噪声（回归）** | $y \leftarrow y + \epsilon,\ \epsilon \sim \mathcal{N}(\mu_e, \sigma_e)$，作用于比例 $r_1$ 的样本 | 训练集 |
| **标签噪声（分类）** | 对比例 $r_1$ 的样本随机翻转标签 | 训练集 |
| **类别失衡** | 多数类 : 少数类重采样至 `imbalance_ratio` | 训练集 |
| **缺失值** | MCAR / MAR / MNAR 掩码，比例 `missing_ratio` | 训练 / 验证 / 测试集 |
| **干净测试标签** | 自然类别分布，无翻转标签 | 测试集 |

## 🔬 消融变体

| 变体 | 开关 | 移除的组件 |
|---------|--------|-----------------|
| **BIAM**（完整版） | — | — |
| **BIAM-B** | `use_bilevel = False` | 双层重加权 → 均匀样本权重 |
| **BIAM-I** | `use_missing_interactions = False` | 缺失 × 特征交互项 |
| **BIAM-H** | `basis_type = 'piecewise_constant'` | 分段线性 hinge 基 → 分段常数基 |

## 📁 实验结果

以下结果来自项目自带的仿真测试套件（`tests/test_biam_simulation.py`，合成数据，CPU，随机种子 {42, 7}）。

### 回归 —— 测试 MSE（越低越好）

| 设置 | BIAM | BIAM-B（无双层优化） |
|---------|------|---------------------|
| 干净数据 | 0.147 | 0.144 |
| 30% 噪声 + 20% 缺失（多种子平均） | **0.436** | 0.457 |

在腐蚀数据下，双层重加权稳定胜出；而在干净数据上两种变体收敛到同一水平——与论文 Table 2 的行为一致。

### 学习到的样本权重的效果

学习到的 ν 正确降低了腐蚀样本的权重：ν 与逐样本损失的 Spearman 相关系数达到 **−1.0**；在腐蚀数据上经双层重加权训练的模型取得 **MSE 0.095**，而均值预测器为 **1.27**。

### 分类

| 设置 | Macro-F1 |
|---------|----------|
| 干净数据（60 轮） | 0.846 |

### 形状函数外推能力（BIAM-H 消融）

训练域 $x_0 \in [-1, 0.8]$，测试域 $x_0 \in [1.0, 1.5]$，真实函数 $y = 3x_0$：

| 基函数 | 测试 MSE |
|-------|----------|
| 分段线性 hinge（BIAM） | **0.12** |
| 分段常数（BIAM-H） | 3.58 |

基于 hinge 的形状函数外推能力好约 30 倍，直接印证了论文 §5 的结论。

> **说明**：在特征独立的玩具合成数据上，缺失交互模块（BIAM vs. BIAM-I）表现出非劣性与明确的模块激活（$|\beta^{\text{miss}}| > 10^{-3}$）；论文中报告的显著差距来自特征高度相关的真实数据集（如 ADNI）。

## ⚙️ 配置说明

所有行为均由 `utils/biam_config.py` 控制（可通过命令行参数覆盖）。关键选项：

| 选项 | 默认值 | 说明 |
|--------|---------|-------------|
| `task` | `'regression'` | `'regression'`（回归）或 `'classification'`（分类） |
| `n_samples` / `n_features` | 1200 / 10 | 合成数据集的规模与维度 |
| `noise_ratio` | 0.3 | 被腐蚀的训练样本比例 |
| `noise_mean` / `noise_std` | 1.0 / 0.5 | 加性回归噪声的 $\mathcal{N}(\mu_e, \sigma_e)$ |
| `imbalance_ratio` | 0.2 | 训练集重采样的少数 : 多数类比 |
| `missing_ratio` | 0.2 | 被掩码条目的比例 |
| `missing_mechanism` | `'MCAR'` | `'MCAR'`、`'MAR'` 或 `'MNAR'` |
| `n_knots` | 8 | 每个特征的 hinge 节点数（置于分位数处） |
| `basis_type` | `'piecewise_linear'` | `'piecewise_linear'` 或 `'piecewise_constant'`（BIAM-H） |
| `use_bilevel` | `True` | `False` 关闭双层重加权（BIAM-B） |
| `use_missing_interactions` | `True` | `False` 移除缺失 × 特征交互（BIAM-I） |
| `lambda_l2` / `lambda_l0` | 1e-3 / 1e-4 | 形状函数的平滑 / 稀疏惩罚 |
| `upper_lr` / `lower_lr` | 0.1 / 0.05 | 上层（θ）与下层（w）学习率 |
| `epochs` / `batch_size` | 200 / 64 | 训练安排 |
| `seed` | 42 | 全局随机种子 |

## 🧪 测试

项目自带覆盖单元、功能、仿真与性能四个层级的 63 个测试用例：

```bash
# 全量测试
python -m pytest tests/ -q

# 按层级运行
python -m pytest tests/test_biam_models.py -v        # 模型内部（hinge 基、缺失处理、正则）
python -m pytest tests/test_biam_data.py -v          # 腐蚀管线（噪声 / 失衡 / 缺失）
python -m pytest tests/test_biam_optimizer.py -v     # 双层优化正确性
python -m pytest tests/test_biam_simulation.py -v    # 端到端行为 vs. 论文结论
python -m pytest tests/test_biam_performance.py -v   # 可扩展性

# 或使用自带运行器
python run_tests.py
```

## ☑️ 待办清单

- [ ] 真实数据集基准测试（ADNI、UCI Adult、Credit）
- [ ] GPU 加速与混合精度训练
- [ ] 交互项的形状函数可视化
- [ ] 双层学习率的自动超参搜索

## 🪪 许可证

本项目基于 MIT 许可证发布 —— 详见 [LICENSE](./LICENSE) 文件。
