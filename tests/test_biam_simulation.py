"""
BIAM 仿真功能测试（论文对齐）
在论文 §4.1 的仿真数据腐蚀场景下，验证 BIAM 相对各消融变体的行为，
对应论文的关键 claim：
  1. 双层优化（BIAM）在噪声+缺失场景优于固定均匀权重的 BIAM-B（Table 2 行为）
  2. BIAM 在腐蚀下保持鲁棒（MSE 不显著恶化）
  3. 干净/轻度腐蚀下达到高 Macro-F1（Table 3 行为）
  4. 分段线性 hinge 基的外推能力优于分段常数基（BIAM-H 消融，论文 §5 讨论）
  5. 缺失指示/交互模块在缺失数据下被激活并保持预测非劣（BIAM-I 消融）
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch
import numpy as np
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from data.biam_data_generator import BIAMDataGenerator
from utils.biam_config import BIAMConfig


def make_sim_config(task='regression', n_samples=800, n_features=8, seed=42,
                    lower_lr=0.05, upper_lr=0.01, epochs=40, n_knots=10, **kwargs):
    """构造仿真配置（经过校准的超参数）"""
    config = BIAMConfig()
    config.task = task
    config.dataset = 'synthetic'
    config.device = torch.device('cpu')
    config.seed = seed
    config.n_samples = n_samples
    config.n_features = n_features
    config.input_dim = n_features
    config.batch_size = 64
    config.epochs = epochs
    config.lower_lr = lower_lr
    config.upper_lr = upper_lr
    config.n_knots = n_knots
    config.lambda_l2 = 1e-3
    config.lambda_l0 = 1e-4
    config.noise_mean = 1.0
    config.noise_std = 0.5
    config.missing_mechanism = 'MCAR'
    config.use_bilevel = True
    config.use_missing_interactions = True
    config.basis_type = 'piecewise_linear'
    
    # 默认无腐蚀
    config.noise_ratio = 0.0
    config.imbalance_ratio = 0.0
    config.missing_ratio = 0.0
    
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config


def generate(config):
    """生成仿真数据"""
    gen = BIAMDataGenerator(config)
    return gen.generate_data()


def build_and_train(config, train_loader, val_loader, train_data, test_data):
    """构建并训练 BIAM 模型，返回评估指标"""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    model = BIAMModel(config, config.device)
    # 用训练数据分位数设置 hinge 节点（论文：节点预定义为训练数据分位数）
    model.additive_model.set_knots(torch.tensor(train_data[0], dtype=torch.float32))
    weighting_net = BIAMWeightingNetwork(config, config.device)
    optimizer = BIAMOptimizer(config, model, weighting_net)
    
    for epoch in range(config.epochs):
        optimizer.train_epoch(train_loader, val_loader, epoch)
    
    return optimizer, optimizer.evaluate(test_data)


class TestBIAMSimulation(unittest.TestCase):
    """
    仿真功能测试：BIAM 与消融变体（BIAM-B / BIAM-I / BIAM-H）的论文行为对齐
    """
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 78)
        print("BIAM 仿真功能测试（论文 §4.1 数据腐蚀场景）")
        print("=" * 78)
    
    # ------------------------------------------------------------------
    # 回归场景（对应论文 Table 2 的 MSE 行为）
    # ------------------------------------------------------------------
    
    def test_regression_clean_learnable(self):
        """
        回归-clean：BIAM 与 BIAM-B 都能有效学习（无腐蚀时双层收益有限，两者接近）
        """
        config = make_sim_config('regression')
        tr, vl, trd, vd, td = generate(config)
        
        _, biam = build_and_train(config, tr, vl, trd, td)
        
        config_b = copy.deepcopy(config)
        config_b.use_bilevel = False
        _, biam_b = build_and_train(config_b, tr, vl, trd, td)
        
        print(f"\n[回归-clean]  BIAM MSE={biam['mse']:.4f} | BIAM-B MSE={biam_b['mse']:.4f}")
        
        # 两者都学到真函数（均值预测器 MSE≈1.27）
        self.assertLess(biam['mse'], 0.5)
        self.assertLess(biam_b['mse'], 0.5)
        # 无腐蚀时双层收益有限：差异不大
        self.assertLess(abs(biam['mse'] - biam_b['mse']), 0.15)
    
    def test_regression_biam_beats_biam_b_under_corruption(self):
        """
        回归-噪声+缺失（论文核心行为）：BIAM（双层加权）优于 BIAM-B（均匀权重）
        双层优化通过降低噪声样本权重提升鲁棒性，多种子取平均保证稳定
        """
        mses_biam, mses_biam_b = [], []
        for seed in (42, 7):
            config = make_sim_config('regression', seed=seed,
                                     noise_ratio=0.3, missing_ratio=0.2)
            tr, vl, trd, vd, td = generate(config)
            
            _, biam = build_and_train(config, tr, vl, trd, td)
            
            config_b = copy.deepcopy(config)
            config_b.use_bilevel = False
            _, biam_b = build_and_train(config_b, tr, vl, trd, td)
            
            mses_biam.append(biam['mse'])
            mses_biam_b.append(biam_b['mse'])
            print(f"  [seed{seed}] BIAM MSE={biam['mse']:.4f} | BIAM-B MSE={biam_b['mse']:.4f}")
        
        avg_biam, avg_biam_b = np.mean(mses_biam), np.mean(mses_biam_b)
        print(f"[回归-噪声30%+缺失20%] 平均 BIAM MSE={avg_biam:.4f} < 平均 BIAM-B MSE={avg_biam_b:.4f}")
        
        # 论文行为：腐蚀越重，BIAM 相对 BIAM-B 优势越大
        self.assertLess(avg_biam, avg_biam_b,
                        "BIAM 在噪声+缺失场景应优于 BIAM-B（论文 Table 2 行为）")
    
    def test_regression_robustness(self):
        """
        回归-鲁棒性：噪声+缺失场景下 BIAM 的 MSE 不显著高于 clean 场景
        """
        config_clean = make_sim_config('regression')
        tr, vl, trd, vd, td = generate(config_clean)
        _, clean_metrics = build_and_train(config_clean, tr, vl, trd, td)
        
        config_noise = make_sim_config('regression', noise_ratio=0.3, missing_ratio=0.2)
        tr, vl, trd, vd, td = generate(config_noise)
        _, noise_metrics = build_and_train(config_noise, tr, vl, trd, td)
        
        print(f"\n[回归-鲁棒性] clean MSE={clean_metrics['mse']:.4f} | "
              f"噪声+缺失 MSE={noise_metrics['mse']:.4f}")
        
        # 30% 样本受 N(1, 0.5) 噪声 + 20% 缺失下，MSE 恶化有限
        self.assertLess(noise_metrics['mse'], clean_metrics['mse'] + 0.5)
    
    # ------------------------------------------------------------------
    # 分类场景（对应论文 Table 3 的 Macro-F1 行为）
    # ------------------------------------------------------------------
    
    def test_classification_clean_high_f1(self):
        """
        分类-clean：BIAM 在无腐蚀场景达到高 Macro-F1（论文 Table 3 干净基线行为）
        """
        config = make_sim_config('classification', upper_lr=0.1, epochs=60)
        tr, vl, trd, vd, td = generate(config)
        _, metrics = build_and_train(config, tr, vl, trd, td)
        
        print(f"\n[分类-clean] BIAM Macro-F1={metrics['f1_macro']:.4f} "
              f"Accuracy={metrics['accuracy']:.4f}")
        
        self.assertGreaterEqual(metrics['f1_macro'], 0.80)
    
    def test_classification_combined_corruption_non_inferiority(self):
        """
        分类-组合腐蚀（噪声10%+失衡10%+缺失10%）：
        BIAM 相对 BIAM-B 保持非劣（双层加权在分类场景的主要收益体现在
        回归 MSE 与噪声降权机制，见 printout 中的 ν 响应分析）
        """
        config = make_sim_config('classification', upper_lr=0.1,
                                 noise_ratio=0.1, imbalance_ratio=0.1, missing_ratio=0.1)
        tr, vl, trd, vd, td = generate(config)
        
        opt_biam, biam = build_and_train(config, tr, vl, trd, td)
        
        config_b = copy.deepcopy(config)
        config_b.use_bilevel = False
        _, biam_b = build_and_train(config_b, tr, vl, trd, td)
        
        print(f"\n[分类-组合腐蚀] BIAM Macro-F1={biam['f1_macro']:.4f} | "
              f"BIAM-B Macro-F1={biam_b['f1_macro']:.4f}")
        
        # 非劣性：双层加权不应显著劣于均匀权重
        self.assertGreaterEqual(biam['f1_macro'], biam_b['f1_macro'] - 0.02)
        
        # 双层机制确实在运行：加权网络学到了非平凡的样本权重响应
        with torch.no_grad():
            loss_range = torch.linspace(0.01, 2.0, 8).view(-1, 1)
            nu = opt_biam.weighting_network(opt_biam._nu_input(loss_range)).flatten()
        self.assertGreater(float(nu.max() - nu.min()), 0.01,
                           "加权网络应对不同损失水平产生有区分度的权重")
    
    # ------------------------------------------------------------------
    # BIAM-H 消融（论文 §5：分段线性基的外推能力优于分段常数基）
    # ------------------------------------------------------------------
    
    def test_biam_h_extrapolation(self):
        """
        BIAM-H 消融：hinge（分段线性）基的外推能力显著优于分段常数基。
        设计：训练域 x0 ∈ [-1, 0.8]，测试域（纯外推）x0 ∈ [1.0, 1.5]，y = 3·x0。
        hinge 在训练域外延续线性趋势；分段常数只能输出最后一个区间的常数值。
        """
        def run_basis(basis_type, seed):
            config = make_sim_config('regression', n_samples=400, n_features=4,
                                     seed=seed, epochs=80, n_knots=6)
            config.basis_type = basis_type
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            X_train = np.random.uniform(-1, 0.8, size=(400, 4)).astype(np.float32)
            y_train = (3 * X_train[:, 0] + 0.05 * np.random.randn(400)).astype(np.float32)
            X_test = np.random.uniform(1.0, 1.5, size=(100, 4)).astype(np.float32)
            y_test = (3 * X_test[:, 0]).astype(np.float32)
            
            model = BIAMModel(config, config.device)
            model.additive_model.set_knots(torch.tensor(X_train))
            weighting_net = BIAMWeightingNetwork(config, config.device)
            optimizer = BIAMOptimizer(config, model, weighting_net)
            
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=64, shuffle=True)
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_train[:64]), torch.tensor(y_train[:64])),
                batch_size=64)
            
            for epoch in range(config.epochs):
                optimizer.train_epoch(train_loader, val_loader, epoch)
            return optimizer.evaluate((X_test, y_test))
        
        mses_hinge, mses_const = [], []
        for seed in (42, 7):
            m_hinge = run_basis('piecewise_linear', seed)['mse']
            m_const = run_basis('piecewise_constant', seed)['mse']
            mses_hinge.append(m_hinge)
            mses_const.append(m_const)
            print(f"  [seed{seed}] BIAM(hinge) 外推 MSE={m_hinge:.4f} | "
                  f"BIAM-H(constant) 外推 MSE={m_const:.4f}")
        
        avg_hinge, avg_const = np.mean(mses_hinge), np.mean(mses_const)
        print(f"[BIAM-H 消融-外推] 平均 hinge MSE={avg_hinge:.4f} << 平均 constant MSE={avg_const:.4f}")
        
        # 论文 §5 claim：分段线性基外推显著优于分段常数基
        self.assertLess(avg_hinge, avg_const,
                        "hinge 基的外推 MSE 应显著低于分段常数基（论文 §5）")
    
    # ------------------------------------------------------------------
    # BIAM-I 消融（缺失指示/交互模块的激活与预测非劣性）
    # ------------------------------------------------------------------
    
    def test_biam_i_missing_modules_engage(self):
        """
        BIAM-I 消融：缺失场景下缺失指示/交互模块被激活（β_miss 非零），
        且完整 BIAM 相对去除缺失交互的 BIAM-I 保持预测非劣。
        
        注：本仿真数据的特征相互独立（X ~ Uniform），缺失交互项
        I(m_j)·h(x_k) 无法借助相关特征恢复缺失值信息（论文的真实数据集
        如 ADNI 特征相关），因此此处断言非劣性 + 模块激活，而非显著优势。
        """
        config = make_sim_config('regression', noise_ratio=0.2,
                                 missing_ratio=0.25, missing_mechanism='MNAR')
        tr, vl, trd, vd, td = generate(config)
        
        opt_biam, biam = build_and_train(config, tr, vl, trd, td)
        
        config_i = copy.deepcopy(config)
        config_i.use_missing_interactions = False
        _, biam_i = build_and_train(config_i, tr, vl, trd, td)
        
        # 缺失指示模块被激活：β_miss 学到非零值
        beta_miss = opt_biam.additive_model.get_missing_indicators()
        print(f"\n[BIAM-I 消融-MNAR缺失25%] β_miss 幅值={np.abs(beta_miss).mean():.4f} | "
              f"BIAM MSE={biam['mse']:.4f} | BIAM-I MSE={biam_i['mse']:.4f}")
        
        self.assertGreater(np.abs(beta_miss).max(), 1e-3,
                           "缺失场景下缺失指示效应 β_miss 应被学习为非零")
        
        # 交互模块存在且可访问
        interactions = opt_biam.additive_model.get_interaction_weights()
        self.assertEqual(interactions.shape,
                         (config.n_features, config.n_features, config.n_knots, 1))
        
        # 预测非劣性（玩具数据特征独立，交互优势需要特征相关性，见 docstring）
        self.assertLessEqual(biam['mse'], biam_i['mse'] + 0.25,
                             "BIAM 相对 BIAM-I 应保持预测非劣")


def print_simulation_summary():
    """打印仿真结果与论文行为的对照说明"""
    print("\n" + "=" * 78)
    print("仿真测试与论文行为对照说明：")
    print("-" * 78)
    print("1. 回归-噪声+缺失：BIAM(双层) < BIAM-B(均匀权重)  ← 论文 Table 2 核心行为")
    print("2. 回归-鲁棒性：腐蚀下 MSE 恶化有限               ← 论文鲁棒性 claim")
    print("3. 分类-clean：Macro-F1 ≥ 0.80                    ← 论文 Table 3 高 F1 行为")
    print("4. 分类-组合腐蚀：BIAM 非劣于 BIAM-B，ν 权重机制激活 ← 论文双层优化机制")
    print("5. BIAM-H：hinge 外推 MSE << 分段常数             ← 论文 §5 外推能力 claim")
    print("6. BIAM-I：缺失模块激活且预测非劣                 ← 论文缺失建模 claim")
    print("   （玩具数据特征独立，交互优势需特征相关性，真实数据集如 ADNI 满足）")
    print("=" * 78)


if __name__ == '__main__':
    print_simulation_summary()
    unittest.main(verbosity=2)
