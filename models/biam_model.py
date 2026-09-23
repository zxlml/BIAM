"""
BIAM Model
主模型类：集成加性模型与加权网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np

# 正确导入两个子模块（原代码缺失导致 NameError）
from models.biam_additive_model import BIAMAdditiveModel
from models.biam_weighting_network import BIAMWeightingNetwork

class BIAMModel(nn.Module):
    """
    Main BIAM model that combines additive model with weighting network
    """
    
    def __init__(self, config, device):
        """
        Initialize BIAM model
        
        Args:
            config: BIAM configuration
            device: Device to run on
        """
        super(BIAMModel, self).__init__()
        
        self.config = config
        self.device = device
        self.task = config.task
        
        # Initialize additive model
        self.additive_model = BIAMAdditiveModel(config, device)
        
        # Initialize weighting network
        self.weighting_network = BIAMWeightingNetwork(config, device)
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        """
        前向传播：只返回加性模型的预测
        （样本权重由双层优化器在训练时调用 weighting network 计算）
        """
        predictions = self.additive_model(x)
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from additive model
        """
        return self.additive_model.get_feature_importance()
    
    def get_missing_indicators(self):
        """
        Get missing value indicators
        """
        return self.additive_model.get_missing_indicators()
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        通过对输入加小扰动做 Monte Carlo 估计预测不确定性（NaN 安全）
        """
        predictions = []
        for _ in range(n_samples):
            x_noisy = x + torch.randn_like(x) * 0.01
            pred = self.forward(x_noisy)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        return mean_pred, std_pred
