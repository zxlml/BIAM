"""
BIAM Additive Model
论文 Eq.1 的可解释加性模型：
    f(x, m; w) = β0 + Σ_j f_j(x_j) + Σ_j β_j^miss · m_j + Σ_{j,k,τ} α_{j,k,τ} · I(m_j=1) · h(x_k; η_τ)
    f_j(x_j)   = Σ_τ β_{j,τ} · h(x_j; η_τ),  h(x; η) = max(0, x - η) 为分段线性 hinge 基
节点 η_τ 预定义为训练数据的分位数；缺失特征的主效应置零，通过缺失指示与交互项建模缺失信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple

class BIAMAdditiveModel(nn.Module):
    """
    BIAM 加性模型：主效应(hinge 基) + 缺失指示效应 + 缺失-特征交互效应
    """
    
    def __init__(self, config, device):
        """
        Initialize BIAM additive model
        
        Args:
            config: BIAM configuration
            device: Device to run on
        """
        super(BIAMAdditiveModel, self).__init__()
        
        self.config = config
        self.device = device
        self.task = config.task
        
        # 模型规模参数
        self.input_dim = getattr(config, 'input_dim', 100)
        self.output_dim = 1 if self.task == 'regression' else getattr(config, 'num_classes', 2)
        self.n_knots = getattr(config, 'n_knots', 8)
        self.basis_type = getattr(config, 'basis_type', 'piecewise_linear')
        self.use_missing_interactions = getattr(config, 'use_missing_interactions', True)
        
        self._build_model()
        
        self.to(device)
    
    def _build_model(self):
        """
        构建模型参数：
        - beta0: 截距项（每输出维）
        - B: 主效应系数 β_{j,τ}，形状 (p, L, d)，小随机初始化
        - beta_miss: 缺失指示效应 β_j^miss，形状 (p, d)，零初始化
        - A: 缺失交互系数 α_{j,k,τ}，形状 (p, p, L, d)，零初始化（稀疏起步）
        """
        p, L, d = self.input_dim, self.n_knots, self.output_dim
        
        # 截距 β0
        self.beta0 = nn.Parameter(torch.zeros(d))
        
        # 主效应系数 β_{j,τ}（小随机初始化，保证初始非零梯度）
        self.B = nn.Parameter(torch.randn(p, L, d) * 0.01)
        
        # 缺失指示效应 β_j^miss（零初始化）
        self.beta_miss = nn.Parameter(torch.zeros(p, d))
        
        # 缺失-特征交互系数 α_{j,k,τ}（零初始化 → 天然稀疏，配合 ℓ0 正则剪枝）
        if self.use_missing_interactions:
            self.A = nn.Parameter(torch.zeros(p, p, L, d))
        else:
            self.A = None
        
        # hinge 节点 η_τ，默认 linspace(-1,1,L)，可通过 set_knots 用训练数据分位数覆盖
        knots = torch.linspace(-1.0, 1.0, L)
        self.register_buffer('knots', knots)
        
        # 兼容旧接口：missing_indicators 作为可学习参数（β_j^miss 的别名视图）
        # 旧测试通过 get_missing_indicators() 访问，这里保留参数名以兼容 state_dict
        self._has_knots_data = False
    
    def set_knots(self, X: torch.Tensor):
        """
        用训练数据的分位数设置 hinge 节点 η_τ（论文：节点预定义为训练数据分位数）
        对含缺失值的数据，使用非缺失观测值计算分位数。
        
        Args:
            X: 训练数据张量，形状 (n, p)，可含 NaN
        """
        with torch.no_grad():
            if X.dim() != 2:
                raise ValueError("X must be 2D tensor (n_samples, n_features)")
            
            # 对各特征的观测分位数取平均作为节点
            knots = torch.zeros(self.n_knots, device=X.device)
            n_valid = 0
            for j in range(min(X.shape[1], self.input_dim)):
                col = X[:, j]
                col = col[~torch.isnan(col)]
                if col.numel() == 0:
                    continue
                quantiles = torch.linspace(0, 1, self.n_knots, device=X.device)
                knots += torch.quantile(col, quantiles)
                n_valid += 1
            
            if n_valid > 0:
                knots /= n_valid
            else:
                knots = torch.linspace(-1.0, 1.0, self.n_knots, device=X.device)
            
            self.knots = knots.to(self.device)
            self._has_knots_data = True
    
    def _compute_hinge_basis(self, xf: torch.Tensor) -> torch.Tensor:
        """
        计算 hinge 基 h(x_k; η_τ) = max(0, x - η_τ)
        
        Args:
            xf: 缺失填充后的输入，形状 (b, p)
        
        Returns:
            基函数张量，形状 (b, p, L)
        """
        # (b, p, 1) - (L,) -> (b, p, L)
        basis = xf.unsqueeze(-1) - self.knots.view(1, 1, -1)
        
        if self.basis_type == 'piecewise_linear':
            # h(x;η) = max(0, x-η)（论文默认，BIAM 消融前）
            basis = torch.relu(basis)
        else:
            # piecewise_constant：分段常数基 I(x > η)（BIAM-H 消融）
            basis = (basis > 0).float()
        
        return basis
    
    def forward(self, x):
        """
        前向传播（论文 Eq.1）
        
        Args:
            x: 输入特征，形状 (b, p)，NaN 表示缺失
        
        Returns:
            模型预测，形状 (b, d)
        """
        b, p = x.shape
        
        # 缺失指示 m_j = I(x_j 缺失)
        m = torch.isnan(x).float()
        
        # 缺失值填充为 0（主效应通过 mask 置零，不受填充值影响）
        xf = torch.nan_to_num(x, nan=0.0)
        
        # hinge 基 H[b, j, τ] = h(x_j; η_τ)
        H = self._compute_hinge_basis(xf)
        
        # 缺失特征的主效应置零：缺失时 x 被填充为 0，主效应不应有贡献
        H = H * (1.0 - m).unsqueeze(-1)
        
        # 主效应：β0 + Σ_{j,τ} β_{j,τ} h(x_j; η_τ)
        main = self.beta0.view(1, -1) + torch.einsum('bpl,pld->bd', H, self.B)
        
        # 缺失指示效应：Σ_j β_j^miss m_j
        miss = m @ self.beta_miss
        
        # 缺失-特征交互：Σ_{j,k,τ} α_{j,k,τ} I(m_j=1) h(x_k; η_τ)
        if self.use_missing_interactions and self.A is not None:
            Hf = H.reshape(b, p * self.n_knots)  # (b, p*L)
            # A 形状 (p, p, L, d)，reshape 成 (p, p*L, d) 用于矩阵乘法
            A_flat = self.A.reshape(p, p * self.n_knots, self.output_dim)
            inter = self._missing_interaction(m, Hf, A_flat)
            output = main + miss + inter
        else:
            output = main + miss
        
        return output
    
    def _missing_interaction(self, m: torch.Tensor, Hf: torch.Tensor, A_flat: torch.Tensor) -> torch.Tensor:
        """
        计算缺失交互项（逐输出维，避免大中间量）
        
        Args:
            m: 缺失指示 (b, p)
            Hf: hinge 基展平 (b, p*L)
            A_flat: 交互系数 (p, p*L, d)
        
        Returns:
            交互贡献 (b, d)
        """
        b = m.shape[0]
        inter = torch.zeros(b, self.output_dim, device=m.device)
        for dd in range(self.output_dim):
            # (p, p*L) @ (p*L, b) -> (p, b)：每个缺失特征 j 对应的加权和
            Aj = A_flat[..., dd]  # (p, p*L)
            weighted = (Aj.unsqueeze(1) * Hf.unsqueeze(0)).sum(-1)  # (p, b)
            # Σ_j m[b,j] * weighted[j,b]
            inter[:, dd] = torch.einsum('bj,jb->b', m, weighted)
        return inter
    
    def get_feature_importance(self):
        """
        特征重要度：主效应系数 B 的 L2 范数（按特征聚合）
        
        Returns:
            np.ndarray, 形状 (p,)
        """
        with torch.no_grad():
            # B: (p, L, d) -> 按特征聚合
            importance = torch.norm(self.B, p=2, dim=(1, 2))
            return importance.cpu().numpy()
    
    def get_missing_indicators(self):
        """
        缺失指示效应 β_j^miss（按特征聚合到均值）
        
        Returns:
            np.ndarray, 形状 (p,)
        """
        with torch.no_grad():
            return self.beta_miss.mean(dim=1).detach().cpu().numpy()
    
    def get_interaction_weights(self):
        """
        缺失交互权重 α
        
        Returns:
            np.ndarray, 形状 (p, p, L, d)
        """
        if self.A is None:
            return np.zeros((self.input_dim, self.input_dim, self.n_knots, self.output_dim))
        return self.A.detach().cpu().numpy()
    
    def compute_regularization_loss(self, regularization_type='l2'):
        """
        正则项计算（论文 Eq.2：λ1‖w‖₂² + λ2‖w‖₀）
        
        Args:
            regularization_type: 'l2' | 'l0' | 'l1' | 'group_lasso'
        
        Returns:
            正则损失标量张量
        """
        params = [self.B, self.beta0]
        if self.A is not None:
            params.append(self.A)
        
        if regularization_type == 'l2':
            # ‖w‖₂²（对应 λ1 项）
            reg = torch.tensor(0.0, device=self.device)
            for pw in params:
                reg = reg + torch.sum(pw ** 2)
            return reg
        
        elif regularization_type == 'l0':
            # ‖w‖₀ 的可微替代：Σ (1 - exp(-w²))（对应 λ2 项，促进稀疏）
            reg = torch.tensor(0.0, device=self.device)
            for pw in params:
                reg = reg + torch.sum(1.0 - torch.exp(-pw ** 2))
            return reg
        
        elif regularization_type == 'l1':
            reg = torch.tensor(0.0, device=self.device)
            for pw in params:
                reg = reg + torch.norm(pw, p=1)
            return reg
        
        elif regularization_type == 'group_lasso':
            # 按特征分组的 group lasso：每个特征的 (L, d) 系数组
            reg = torch.tensor(0.0, device=self.device)
            for j in range(self.input_dim):
                reg = reg + torch.norm(self.B[j], p=2)
            return reg
        
        else:
            return torch.tensor(0.0, device=self.device)
    
    def get_penalty(self):
        """
        论文 Eq.2 的完整正则项：λ1‖w‖₂² + λ2‖w‖₀
        """
        lambda_l2 = getattr(self.config, 'lambda_l2', 1e-3)
        lambda_l0 = getattr(self.config, 'lambda_l0', 1e-4)
        return lambda_l2 * self.compute_regularization_loss('l2') + \
               lambda_l0 * self.compute_regularization_loss('l0')
    
    def get_model_interpretation(self, x_sample):
        """
        单样本解释：各特征贡献、缺失效应、交互权重
        
        Args:
            x_sample: 输入样本，形状 (1, p) 或 (p,)
        
        Returns:
            解释字典
        """
        with torch.no_grad():
            if x_sample.dim() == 1:
                x_sample = x_sample.unsqueeze(0)
            
            x_s = x_sample.to(self.device)
            m = torch.isnan(x_s).float()
            xf = torch.nan_to_num(x_s, nan=0.0)
            H = self._compute_hinge_basis(xf) * (1.0 - m).unsqueeze(-1)
            
            # 特征贡献：Σ_τ β_{j,τ} h(x_j; η_τ)，形状 (p, d)
            contrib = torch.einsum('bpl,pld->pd', H, self.B)
            feature_contributions = contrib.mean(dim=1).cpu().numpy()
            
            prediction = self.forward(x_s)
            
            interpretation = {
                'feature_contributions': feature_contributions,
                'missing_indicators': self.get_missing_indicators(),
                'interaction_weights': self.get_interaction_weights(),
                'prediction': prediction.cpu().numpy()
            }
            return interpretation
