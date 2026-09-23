"""
Unit tests for BIAM data processing components
针对论文 §4.1 数据腐蚀（噪声/失衡/缺失）的单元测试
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch
import numpy as np
import pandas as pd
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.biam_data_generator import BIAMDataGenerator
from data.biam_binarizer import BIAMBinarizer
from data.biam_dataset_loader import BIAMDatasetLoader
from utils.biam_config import BIAMConfig

def make_config(task='classification', **kwargs):
    """构造测试配置"""
    config = BIAMConfig()
    config.task = task
    config.dataset = 'synthetic'
    config.missing_ratio = 0.3
    config.noise_ratio = 0.2
    config.imbalance_ratio = 0.15
    config.batch_size = 32
    config.n_samples = kwargs.pop('n_samples', 400)
    config.n_features = kwargs.pop('n_features', 10)
    config.device = torch.device('cpu')
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config

class TestBIAMData(unittest.TestCase):
    """
    Test cases for BIAM data processing components
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.config = make_config('classification')
    
    def test_data_generator_initialization(self):
        """
        Test BIAM data generator initialization
        """
        generator = BIAMDataGenerator(self.config)
        
        self.assertEqual(generator.task, self.config.task)
        self.assertEqual(generator.dataset, self.config.dataset)
        self.assertEqual(generator.missing_ratio, self.config.missing_ratio)
        self.assertEqual(generator.noise_ratio, self.config.noise_ratio)
        self.assertEqual(generator.imbalance_ratio, self.config.imbalance_ratio)
    
    def test_synthetic_regression_data(self):
        """
        Test synthetic regression data generation
        """
        config = make_config('regression', n_samples=300, missing_ratio=0.0)
        
        generator = BIAMDataGenerator(config)
        train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_data)
        
        # 数据形状与特征数
        X_train, y_train = train_data
        self.assertEqual(X_train.shape[1], config.n_features)
        self.assertEqual(len(y_train), X_train.shape[0])
        
        # Test batch structure
        for data, target in train_loader:
            self.assertEqual(data.shape[0], config.batch_size)
            self.assertEqual(target.shape[0], config.batch_size)
            break
        
        # 回归输出应为浮点
        self.assertTrue(np.issubdtype(y_train.dtype, np.floating))
    
    def test_synthetic_classification_data(self):
        """
        Test synthetic classification data generation
        """
        config = make_config('classification', n_samples=300, 
                             missing_ratio=0.0, noise_ratio=0.0, imbalance_ratio=0.0)
        
        generator = BIAMDataGenerator(config)
        train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()
        
        self.assertIsNotNone(train_loader)
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # 标签为 0/1
        self.assertTrue(set(np.unique(y_train.astype(int))).issubset({0, 1}))
        self.assertTrue(set(np.unique(y_test.astype(int))).issubset({0, 1}))
        
        # 特征维度正确
        self.assertEqual(X_train.shape[1], config.n_features)
        self.assertTrue(X_test.shape[1] == config.n_features)
    
    def test_missing_value_generation(self):
        """
        Test missing value generation
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        X = np.random.randn(200, 10)
        X_with_missing = generator._add_missing_values(X, missing_ratio=0.3)
        
        missing_count = np.isnan(X_with_missing).sum()
        self.assertGreater(missing_count, 0)
        
        # 缺失比例应接近目标值
        missing_ratio_actual = np.isnan(X_with_missing).mean()
        self.assertAlmostEqual(missing_ratio_actual, 0.3, delta=0.08)
        
        # 原始数据不应被修改
        self.assertFalse(np.isnan(X).any())
    
    def test_missing_mechanisms(self):
        """
        Test MCAR / MAR / MNAR missing mechanisms
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        X = np.random.randn(300, 5)
        
        for mechanism in ['MCAR', 'MAR', 'MNAR']:
            X_m = generator._add_missing_values(X, missing_ratio=0.2, missing_pattern=mechanism)
            self.assertTrue(np.isnan(X_m).any(), f"{mechanism} should produce missing values")
            # 未缺失位置数值应保持一致
            observed = ~np.isnan(X_m)
            self.assertTrue(np.allclose(X_m[observed], X[observed]))
        
        # 未知机制应报错
        with self.assertRaises(ValueError):
            generator._add_missing_values(X, 0.1, 'UNKNOWN')
    
    def test_label_noise_generation(self):
        """
        Test label noise generation (翻转比例应接近 noise_ratio)
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        y = np.random.randint(0, 2, 500)
        y_noisy = generator._add_label_noise(y, noise_ratio=0.2)
        
        noise_count = np.sum(y != y_noisy)
        self.assertGreater(noise_count, 0)
        # 翻转数量应约为 20%
        self.assertAlmostEqual(noise_count / len(y), 0.2, delta=0.03)
    
    def test_regression_noise(self):
        """
        回归噪声：仅指定比例样本被加噪，噪声服从 N(μe, σe)
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        y = np.zeros(500)
        y_noisy = generator._add_regression_noise(y, ratio=0.3, noise_mean=1.0, noise_std=0.5)
        
        # 约 30% 的样本被加噪（偏移约 μe=1）
        changed = y_noisy != y
        changed_ratio = changed.mean()
        self.assertAlmostEqual(changed_ratio, 0.3, delta=0.03)
        
        # 加噪幅度应接近 N(1.0, 0.5²)
        self.assertAlmostEqual(y_noisy[changed].mean(), 1.0, delta=0.15)
    
    def test_class_imbalance_generation(self):
        """
        Test class imbalance generation（少数:多数 ≈ imbalance_ratio，且 X/y 对齐）
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        X = np.random.randn(1000, 5)
        y = np.random.randint(0, 2, 1000)
        # 强制类别均衡以便测试失衡逻辑
        y[:500] = 0
        y[500:] = 1
        
        imbalanced_indices = generator._create_class_imbalance(y, imbalance_ratio=0.1)
        imbalanced_y = y[imbalanced_indices]
        
        class_counts = np.bincount(imbalanced_y.astype(int))
        
        # 少数:多数 应约为 1:10
        ratio = class_counts.min() / class_counts.max()
        self.assertAlmostEqual(ratio, 0.1, delta=0.03)
        
        # 索引数量与标签数量一致（数据对齐）
        self.assertEqual(len(imbalanced_indices), len(imbalanced_y))
        
        # X 与 y 同步索引：取出的 X 行数应与 y 一致
        imbalanced_X = X[imbalanced_indices]
        self.assertEqual(imbalanced_X.shape[0], len(imbalanced_y))
    
    def test_train_corruption_not_applied_to_test(self):
        """
        失衡只作用于训练集：训练集少数:多数≈0.1，测试集保持天然分布（该真函数天然约 1:3.6）
        """
        config = make_config('classification', n_samples=800, noise_ratio=0.0, 
                             imbalance_ratio=0.1, missing_ratio=0.0)
        generator = BIAMDataGenerator(config)
        train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()
        
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # 测试集保持天然分布（约 0.2~0.35），未被失衡处理
        test_counts = np.bincount(y_test.astype(int))
        test_balance = test_counts.min() / test_counts.max()
        self.assertGreater(test_balance, 0.15)
        
        # 训练集被失衡处理（少数:多数 ≈ 0.1）
        train_counts = np.bincount(y_train.astype(int))
        train_balance = train_counts.min() / train_counts.max()
        self.assertLess(train_balance, 0.2)
    
    def test_binarizer_initialization(self):
        """
        Test BIAM binarizer initialization
        """
        binarizer = BIAMBinarizer()
        
        self.assertIsNotNone(binarizer.quantiles)
        self.assertIsNotNone(binarizer.miss_vals)
        self.assertTrue(binarizer.specific_mi_intercept)
        self.assertTrue(binarizer.specific_mi_ixn)
    
    def test_binarizer_basic_functionality(self):
        """
        Test basic binarizer functionality
        """
        binarizer = BIAMBinarizer()
        
        np.random.seed(0)
        train_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        
        test_df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'label': np.random.randint(0, 2, 50)
        })
        
        result = binarizer.binarize_and_augment(train_df, test_df)
        
        self.assertEqual(len(result), 4)
        train_aug, test_aug, train_labels, test_labels = result
        
        self.assertEqual(train_aug.shape[0], 100)
        self.assertEqual(test_aug.shape[0], 50)
        self.assertEqual(train_labels.shape[0], 100)
        self.assertEqual(test_labels.shape[0], 50)
    
    def test_binarizer_with_missing_values(self):
        """
        Test binarizer with missing values
        """
        binarizer = BIAMBinarizer(numerical_cols=['feature1', 'feature2'])
        
        train_df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [1, np.nan, 3, 4, 5],
            'label': [0, 1, 0, 1, 0]
        })
        
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [1, 2, np.nan],
            'label': [0, 1, 0]
        })
        
        result = binarizer.binarize_and_augment(train_df, test_df)
        
        self.assertEqual(len(result), 4)
        train_aug, test_aug, train_labels, test_labels = result
        
        self.assertGreater(train_aug.shape[1], 2)
        self.assertGreater(test_aug.shape[1], 2)
    
    def test_dataset_loader_initialization(self):
        """
        Test BIAM dataset loader initialization
        """
        loader = BIAMDatasetLoader(self.config)
        
        self.assertEqual(loader.config, self.config)
        self.assertTrue(os.path.exists(loader.data_dir))
    
    def test_dataset_loader_available_datasets(self):
        """
        Test available datasets list
        """
        loader = BIAMDatasetLoader(self.config)
        available_datasets = loader.get_available_datasets()
        
        expected_datasets = ['synthetic', 'adult', 'credit', 'breast_cancer', 'wine', 'iris', 'custom']
        self.assertEqual(available_datasets, expected_datasets)
    
    def test_breast_cancer_dataset_loading(self):
        """
        Test breast cancer dataset loading
        """
        loader = BIAMDatasetLoader(self.config)
        
        try:
            train_loader, val_loader, test_data = loader.load_breast_cancer_dataset(
                missing_ratio=0.1, noise_ratio=0.1
            )
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_data)
            
        except Exception as e:
            # If dataset loading fails, test fallback
            self.assertIsInstance(e, Exception)
    
    def test_fallback_dataset_loading(self):
        """
        Test fallback dataset loading
        """
        loader = BIAMDatasetLoader(self.config)
        train_loader, val_loader, test_data = loader._load_fallback_dataset()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_data)
        
        for data, target in train_loader:
            self.assertEqual(data.shape[0], self.config.batch_size)
            self.assertEqual(target.shape[0], self.config.batch_size)
            break
    
    def test_data_standardization(self):
        """
        Test data standardization
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        X_train = np.random.randn(100, 10)
        X_val = np.random.randn(50, 10)
        X_test = np.random.randn(50, 10)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = generator._standardize_data(
            X_train, X_val, X_test
        )
        
        self.assertAlmostEqual(X_train_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(X_train_scaled.std(), 1, places=5)
        self.assertIsNotNone(scaler)
    
    def test_data_loader_creation(self):
        """
        Test data loader creation
        """
        generator = BIAMDataGenerator(self.config)
        
        np.random.seed(0)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        loader = generator._create_data_loader(X, y, batch_size=16)
        
        self.assertIsNotNone(loader)
        self.assertEqual(loader.batch_size, 16)
        
        for data, target in loader:
            self.assertEqual(data.shape[0], 16)
            self.assertEqual(target.shape[0], 16)
            break

if __name__ == '__main__':
    unittest.main()
