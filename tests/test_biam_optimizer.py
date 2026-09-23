"""
Unit tests for BIAM bilevel optimizer
针对论文 Eq.4-6 双层优化的单元测试
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.biam_model import BIAMModel
from models.biam_additive_model import BIAMAdditiveModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_config import BIAMConfig

def make_config(task='regression', input_dim=5, use_bilevel=True, **kwargs):
    """构造测试配置"""
    config = BIAMConfig()
    config.task = task
    config.input_dim = input_dim
    config.num_classes = 2
    config.device = torch.device('cpu')
    config.n_knots = kwargs.pop('n_knots', 4)
    config.upper_lr = kwargs.pop('upper_lr', 1e-2)
    config.lower_lr = kwargs.pop('lower_lr', 5e-3)
    config.use_bilevel = use_bilevel
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config

def make_data(task='regression', n=64, p=5, with_missing=False, seed=0):
    """构造小规模仿真数据"""
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(n, p)).astype(np.float32)
    if task == 'regression':
        y = (2 * X[:, 0] - X[:, 1] ** 2).astype(np.float32)
    else:
        y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 0.5).astype(np.float32)
    if with_missing:
        mask = np.random.uniform(0, 1, size=X.shape) < 0.15
        X[mask] = np.nan
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    return loader, (X_t, y_t)

class TestBIAMOptimizer(unittest.TestCase):
    """
    Test cases for BIAM bilevel optimizer
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        torch.manual_seed(0)
        np.random.seed(0)
        self.config = make_config('regression')
        self.train_loader, self.train_data = make_data('regression')
        self.val_loader, self.val_data = make_data('regression', seed=1)
        self.test_data = make_data('regression', seed=2)[1]
        
        self.biam_model = BIAMModel(self.config, self.config.device)
        self.weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        self.optimizer = BIAMOptimizer(self.config, self.biam_model, self.weighting_network)
    
    def test_optimizer_initialization(self):
        """
        Test optimizer initialization
        """
        self.assertIsNotNone(self.optimizer.upper_optimizer)
        self.assertIsNotNone(self.optimizer.lower_optimizer)
        self.assertEqual(self.optimizer.upper_lr, self.config.upper_lr)
        self.assertEqual(self.optimizer.lower_lr, self.config.lower_lr)
    
    def test_lower_step_updates_model(self):
        """
        下层更新后加性模型参数应发生变化
        """
        params_before = [p.clone() for p in self.biam_model.additive_model.parameters()]
        
        x, y = self.train_data
        self.optimizer._bilevel_step(x[:16], y[:16], x[16:32], y[16:32])
        
        changed = any(
            not torch.equal(pb, p)
            for pb, p in zip(params_before, self.biam_model.additive_model.parameters())
        )
        self.assertTrue(changed, "下层更新应改变加性模型参数")
    
    def test_bilevel_step_updates_weighting_network(self):
        """
        回归测试（旧代码 bug）：双层步后加权网络参数必须变化（上层学习生效）
        旧实现中 weighting network 输入 detach 导致超梯度断裂
        """
        params_before = [p.clone() for p in self.weighting_network.parameters()]
        
        x, y = self.train_data
        for _ in range(3):
            self.optimizer._bilevel_step(x[:16], y[:16], x[16:32], y[16:32])
        
        changed = any(
            not torch.equal(pb, p)
            for pb, p in zip(params_before, self.weighting_network.parameters())
        )
        self.assertTrue(changed, "双层优化后加权网络参数应被更新（上层梯度不应断裂）")
    
    def test_no_bilevel_keeps_weighting_network_fixed(self):
        """
        BIAM-B 消融（use_bilevel=False）：加权网络不应被更新
        """
        config = make_config('regression', use_bilevel=False)
        biam_model = BIAMModel(config, config.device)
        weighting_network = BIAMWeightingNetwork(config, config.device)
        optimizer = BIAMOptimizer(config, biam_model, weighting_network)
        
        params_before = [p.clone() for p in weighting_network.parameters()]
        
        x, y = self.train_data
        optimizer._bilevel_step(x[:16], y[:16], x[16:32], y[16:32])
        
        unchanged = all(
            torch.equal(pb, p)
            for pb, p in zip(params_before, weighting_network.parameters())
        )
        self.assertTrue(unchanged, "BIAM-B 消融不应更新加权网络")
    
    def test_train_epoch_returns_metrics(self):
        """
        train_epoch 应返回 loss 与 val_loss
        """
        metrics = self.optimizer.train_epoch(self.train_loader, self.val_loader, 0)
        
        self.assertIn('loss', metrics)
        self.assertIn('val_loss', metrics)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertTrue(np.isfinite(metrics['val_loss']))
        
        # 训练历史应记录
        self.assertEqual(len(self.optimizer.get_training_history()['train_loss']), 1)
    
    def test_evaluate_regression_metrics(self):
        """
        回归评估指标：mse/rmse/mae
        """
        metrics = self.optimizer.evaluate(self.test_data)
        
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertTrue(metrics['mse'] >= 0)
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(metrics['mse']), places=6)
    
    def test_evaluate_classification_metrics(self):
        """
        分类评估指标：accuracy/f1_macro（论文指标）/f1_score
        """
        config = make_config('classification')
        biam_model = BIAMModel(config, config.device)
        weighting_network = BIAMWeightingNetwork(config, config.device)
        optimizer = BIAMOptimizer(config, biam_model, weighting_network)
        
        _, test_data = make_data('classification', seed=3)
        metrics = optimizer.evaluate(test_data)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_macro', metrics)
        self.assertIn('f1_score', metrics)
        
        for key in ('accuracy', 'f1_macro', 'f1_score'):
            self.assertTrue(0 <= metrics[key] <= 1)
    
    def test_evaluate_with_missing_values(self):
        """
        测试数据含缺失值时评估不应报错或产生 NaN
        """
        _, test_data = make_data('regression', seed=4, with_missing=True)
        metrics = self.optimizer.evaluate(test_data)
        
        self.assertTrue(np.isfinite(metrics['mse']))
    
    def test_short_training_no_nan(self):
        """
        小规模端到端 3 个 epoch 训练无 NaN
        """
        for epoch in range(3):
            metrics = self.optimizer.train_epoch(self.train_loader, self.val_loader, epoch)
            self.assertTrue(np.isfinite(metrics['loss']))
            self.assertTrue(np.isfinite(metrics['val_loss']))
        
        final = self.optimizer.evaluate(self.test_data)
        self.assertTrue(np.isfinite(final['mse']))
    
    def test_classification_training(self):
        """
        分类任务端到端训练
        """
        config = make_config('classification')
        biam_model = BIAMModel(config, config.device)
        weighting_network = BIAMWeightingNetwork(config, config.device)
        optimizer = BIAMOptimizer(config, biam_model, weighting_network)
        
        train_loader, _ = make_data('classification', seed=5)
        val_loader, _ = make_data('classification', seed=6)
        _, test_data = make_data('classification', seed=7)
        
        for epoch in range(3):
            metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
            self.assertTrue(np.isfinite(metrics['loss']))
        
        final = optimizer.evaluate(test_data)
        self.assertTrue(0 <= final['accuracy'] <= 1)

if __name__ == '__main__':
    unittest.main()
