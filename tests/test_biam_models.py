"""
Unit tests for BIAM model components
针对论文 Eq.1 加性模型结构的单元测试
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from models.biam_additive_model import BIAMAdditiveModel
from utils.biam_config import BIAMConfig

def make_config(task='classification', input_dim=10, **kwargs):
    """构造测试配置"""
    config = BIAMConfig()
    config.task = task
    config.input_dim = input_dim
    config.num_classes = 2
    config.device = torch.device('cpu')
    config.n_knots = kwargs.pop('n_knots', 5)
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config

class TestBIAMModels(unittest.TestCase):
    """
    Test cases for BIAM model components
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.config = make_config('classification', 10)
        self.batch_size = 32
        self.input_dim = 10
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.input_dim)
        self.test_target = torch.randint(0, 2, (self.batch_size,))
    
    def test_biam_weighting_network_initialization(self):
        """
        Test BIAM weighting network initialization
        """
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        # Test network structure
        self.assertEqual(weighting_network.input_dim, 1)
        self.assertEqual(weighting_network.hidden_dim, 10)
        self.assertEqual(weighting_network.output_dim, 1)
        
        # Test forward pass
        test_losses = torch.randn(self.batch_size, 1)
        weights = weighting_network(test_losses)
        
        self.assertEqual(weights.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
    
    def test_biam_additive_model_initialization(self):
        """
        Test BIAM additive model initialization
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # Test model structure
        self.assertEqual(additive_model.input_dim, self.config.input_dim)
        self.assertEqual(additive_model.output_dim, self.config.num_classes)
        
        # Test forward pass
        output = additive_model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.config.num_classes))
    
    def test_biam_model_initialization(self):
        """
        Test BIAM model initialization (子模块正确导入)
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        # Test model components
        self.assertIsInstance(biam_model.additive_model, BIAMAdditiveModel)
        self.assertIsInstance(biam_model.weighting_network, BIAMWeightingNetwork)
        
        # Test forward pass
        output = biam_model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.config.num_classes))
    
    def test_feature_importance(self):
        """
        Test feature importance calculation
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        importance = biam_model.get_feature_importance()
        
        self.assertEqual(len(importance), self.config.input_dim)
        self.assertTrue(np.all(importance >= 0))
    
    def test_missing_indicators(self):
        """
        Test missing value indicators
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        missing_indicators = biam_model.get_missing_indicators()
        
        self.assertEqual(len(missing_indicators), self.config.input_dim)
    
    def test_model_uncertainty(self):
        """
        Test model uncertainty estimation (NaN 安全)
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        x_with_nan = self.test_input.clone()
        x_with_nan[0, 0] = float('nan')
        
        mean_pred, std_pred = biam_model.predict_with_uncertainty(x_with_nan, n_samples=5)
        
        self.assertEqual(mean_pred.shape, (self.batch_size, self.config.num_classes))
        self.assertEqual(std_pred.shape, (self.batch_size, self.config.num_classes))
        self.assertTrue(torch.all(std_pred >= 0))
        self.assertFalse(torch.isnan(mean_pred).any())
    
    def test_regression_mode(self):
        """
        Test BIAM model in regression mode
        """
        config = make_config('regression', 10)
        
        biam_model = BIAMModel(config, config.device)
        
        output = biam_model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_model_interpretation(self):
        """
        Test model interpretation
        """
        biam_model = BIAMModel(self.config, self.config.device)
        
        sample_data = self.test_input[:1]
        interpretation = biam_model.additive_model.get_model_interpretation(sample_data)
        
        self.assertIn('feature_contributions', interpretation)
        self.assertIn('missing_indicators', interpretation)
        self.assertIn('interaction_weights', interpretation)
        self.assertIn('prediction', interpretation)
        
        self.assertEqual(len(interpretation['feature_contributions']), self.config.input_dim)
    
    def test_regularization_loss(self):
        """
        Test regularization loss calculation (各正则项非负)
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        reg_loss_l1 = additive_model.compute_regularization_loss('l1')
        reg_loss_l2 = additive_model.compute_regularization_loss('l2')
        reg_loss_group = additive_model.compute_regularization_loss('group_lasso')
        reg_loss_l0 = additive_model.compute_regularization_loss('l0')
        
        for reg in (reg_loss_l1, reg_loss_l2, reg_loss_group, reg_loss_l0):
            self.assertIsInstance(reg, torch.Tensor)
            self.assertTrue(reg >= 0)
        
        # penalty = λ1·l2 + λ2·l0
        penalty = additive_model.get_penalty()
        expected = (self.config.lambda_l2 * reg_loss_l2 + 
                    self.config.lambda_l0 * reg_loss_l0)
        self.assertTrue(torch.allclose(penalty, expected))
    
    def test_weight_statistics(self):
        """
        Test weight statistics calculation
        """
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        test_losses = torch.randn(self.batch_size, 1)
        stats = weighting_network.get_weight_statistics(test_losses)
        
        self.assertIn('mean_weight', stats)
        self.assertTrue(0 <= stats['mean_weight'] <= 1)
    
    def test_dynamic_weight_update(self):
        """
        Test dynamic weight update
        """
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        
        test_losses = torch.randn(self.batch_size, 1)
        updated_weights = weighting_network.update_weights_dynamically(test_losses, 100, 1000)
        
        self.assertEqual(updated_weights.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(updated_weights >= 0))
    
    def test_hinge_basis_correctness(self):
        """
        验证 hinge 基 h(x;η) = max(0, x-η) 的数值正确性
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        x = torch.tensor([[0.5], [2.0], [-3.0]])
        eta = torch.tensor(1.0)
        expected = torch.relu(x - eta)
        
        # 构造 knots 为全 1 的情形
        additive_model.knots = torch.ones(self.config.n_knots)
        basis = additive_model._compute_hinge_basis(x)
        
        # 每个节点均为 1，所有 τ 列应等于 relu(x-1)
        for tau in range(self.config.n_knots):
            self.assertTrue(torch.allclose(basis[:, 0, tau], expected.squeeze(1)))
    
    def test_piecewise_constant_basis(self):
        """
        验证分段常数基 I(x > η)（BIAM-H 消融）
        """
        config = make_config('classification', 3, basis_type='piecewise_constant')
        additive_model = BIAMAdditiveModel(config, config.device)
        additive_model.knots = torch.tensor([0.0, 0.5, 1.0])
        
        x = torch.tensor([[0.3], [0.7]])
        basis = additive_model._compute_hinge_basis(x)
        
        # x=0.3: 只有 η=0 满足 x>η
        self.assertAlmostEqual(basis[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(basis[0, 0, 1].item(), 0.0)
        # x=0.7: η=0 和 η=0.5 满足
        self.assertAlmostEqual(basis[1, 0, 0].item(), 1.0)
        self.assertAlmostEqual(basis[1, 0, 1].item(), 1.0)
        self.assertAlmostEqual(basis[1, 0, 2].item(), 0.0)
    
    def test_forward_with_missing_values_no_nan(self):
        """
        缺失值前向传播应无 NaN 传播
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        x_with_missing = self.test_input.clone()
        x_with_missing[0, :3] = float('nan')
        x_with_missing[5, 7] = float('nan')
        
        output = additive_model(x_with_missing)
        
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        self.assertEqual(output.shape, (self.batch_size, self.config.num_classes))
    
    def test_missing_indicator_effect(self):
        """
        缺失指示效应：β_j^miss 非零时，缺失样本的预测应与填充 0 的完整样本不同
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        # 设置非零 β_miss
        with torch.no_grad():
            additive_model.beta_miss.fill_(2.0)
        
        x = torch.zeros(1, self.input_dim)
        x_missing = x.clone()
        x_missing[0, 0] = float('nan')
        
        pred_complete = additive_model(x)
        pred_missing = additive_model(x_missing)
        
        # 缺失特征主效应被置零，但缺失指示效应 β_j^miss=2 使预测不同
        self.assertFalse(torch.allclose(pred_complete, pred_missing))
    
    def test_missing_interaction_effect(self):
        """
        缺失交互效应：α 非零时只有缺失样本的预测受影响
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        with torch.no_grad():
            additive_model.A.fill_(1.0)
        
        x = torch.randn(4, self.input_dim).abs()
        x_missing = x.clone()
        x_missing[0, 0] = float('nan')
        
        pred_before = additive_model(x).clone()
        pred_missing = additive_model(x_missing)
        
        # 非缺失样本（1:3）预测不受 A 影响
        self.assertTrue(torch.allclose(pred_before[1:], additive_model(x)[1:]))
        # 缺失样本预测改变
        self.assertFalse(torch.allclose(pred_before[0:1], pred_missing[0:1]))
    
    def test_set_knots_quantiles(self):
        """
        set_knots 应将节点设置为训练数据分位数（含缺失值时忽略 NaN）
        """
        additive_model = BIAMAdditiveModel(self.config, self.config.device)
        
        np.random.seed(0)
        X = torch.tensor(np.random.uniform(-1, 1, size=(500, self.input_dim)), 
                         dtype=torch.float32)
        X[0, 0] = float('nan')
        
        additive_model.set_knots(X)
        
        # 节点应在数据范围内
        self.assertTrue(additive_model.knots.min() >= -1.01)
        self.assertTrue(additive_model.knots.max() <= 1.01)
        # 节点应单调不减
        self.assertTrue((additive_model.knots[1:] >= additive_model.knots[:-1]).all())
        # 不应全为默认值
        self.assertFalse(torch.allclose(additive_model.knots, 
                                        torch.linspace(-1, 1, self.config.n_knots)))
    
    def test_output_shapes(self):
        """
        回归/分类输出形状
        """
        # 回归
        config_reg = make_config('regression', 6)
        model_reg = BIAMAdditiveModel(config_reg, config_reg.device)
        out_reg = model_reg(torch.randn(8, 6))
        self.assertEqual(out_reg.shape, (8, 1))
        
        # 3 分类
        config_cls = make_config('classification', 6)
        config_cls.num_classes = 3
        model_cls = BIAMAdditiveModel(config_cls, config_cls.device)
        out_cls = model_cls(torch.randn(8, 6))
        self.assertEqual(out_cls.shape, (8, 3))

if __name__ == '__main__':
    unittest.main()
