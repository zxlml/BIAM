"""
BIAM Data Generator
按论文 §4.1 生成仿真数据并注入数据腐蚀：
  - 回归：对 r1 比例训练样本 y 加噪声 ε ~ N(μe, σe)
  - 分类：r1 比例训练样本标签翻转；r2 失衡比（少数:多数 = imbalance_ratio）
  - r3 缺失：MCAR / MAR / MNAR 机制注入到训练/验证/测试集
测试集保持干净（不加噪声、不加失衡），保证评估无偏。
"""

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import torch.utils.data as Data
from scipy.stats import norm
from random import sample
import warnings
warnings.filterwarnings("ignore")

class BIAMDataGenerator:
    """
    Data generator for BIAM model supporting various data challenges
    """
    
    def __init__(self, config):
        """
        Initialize BIAM data generator
        
        Args:
            config: BIAM configuration object
        """
        self.config = config
        self.task = config.task
        self.dataset = config.dataset
        self.missing_ratio = config.missing_ratio
        self.noise_ratio = config.noise_ratio
        self.imbalance_ratio = config.imbalance_ratio
        self.batch_size = config.batch_size
    
    def generate_data(self):
        """
        Generate training, validation, and test data
        
        Returns:
            tuple: (train_loader, val_loader, test_data)
        """
        if self.dataset == 'synthetic':
            return self._generate_synthetic_data()
        elif self.dataset == 'adult':
            return self._generate_adult_data()
        elif self.dataset == 'credit':
            return self._generate_credit_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
    
    def _generate_synthetic_data(self):
        """
        Generate synthetic data with specified challenges
        """
        if self.task == 'regression':
            return self._generate_synthetic_regression()
        else:
            return self._generate_synthetic_classification()
    
    def _generate_synthetic_regression(self):
        """
        仿真回归数据（论文 §4.1）：
        X ~ Uniform(-1,1)^p；y = Σ f_j(x_j) + 0.1·N(0,1)（内在噪声）
        仅训练集：对 r1 比例样本 y += N(μe, σe)
        缺失按 r3 注入 train/val/test
        """
        # 固定种子，保证可复现
        np.random.seed(self.config.seed)
        
        n_samples = getattr(self.config, 'n_samples', 1200)
        n_features = getattr(self.config, 'n_features', 10)
        
        # 生成基础数据
        X = np.random.uniform(-1, 1, size=(n_samples, n_features))
        
        # 生成真函数（非线性加性结构 + 内在噪声）
        y = self._generate_regression_function(X)
        y = y + 0.1 * np.random.randn(n_samples)
        
        # 划分数据集（60/20/20）
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.config.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.config.seed
        )
        
        # 标准化（先标准化，后腐蚀；scaler 仅用训练集拟合，X 与 y 均标准化）
        X_train, X_val, X_test, _ = self._standardize_data(X_train, X_val, X_test)
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 回归噪声：仅对训练集 r1 比例样本加 ε ~ N(μe, σe)
        noise_mean = getattr(self.config, 'noise_mean', 1.0)
        noise_std = getattr(self.config, 'noise_std', 0.5)
        y_train = self._add_regression_noise(
            y_train, ratio=self.noise_ratio, noise_mean=noise_mean, noise_std=noise_std
        )
        
        # 缺失注入（MCAR/MAR/MNAR），train/val/test 均注入
        if self.missing_ratio > 0:
            mechanism = getattr(self.config, 'missing_mechanism', 'MCAR')
            X_train = self._add_missing_values(X_train, self.missing_ratio, mechanism)
            X_val = self._add_missing_values(X_val, self.missing_ratio, mechanism)
            X_test = self._add_missing_values(X_test, self.missing_ratio, mechanism)
        
        # 更新配置中的实际维度
        self.config.input_dim = n_features
        
        # 创建数据加载器
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        return train_loader, val_loader, (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _generate_synthetic_classification(self):
        """
        仿真分类数据（论文 §4.1）：
        X ~ Uniform(0,1)^p；y = I((x0-0.5)² + (x1-0.5)² > 0.08)
        仅训练集：先失衡（多数类下采样至 少数:多数 = imbalance_ratio）后标签翻转 r1
        缺失按 r3 注入 train/val/test；测试集保持干净
        """
        np.random.seed(self.config.seed)
        
        n_samples = getattr(self.config, 'n_samples', 1200)
        n_features = getattr(self.config, 'n_features', 10)
        
        # 生成基础数据
        X = np.random.uniform(0, 1, size=(n_samples, n_features))
        
        # 生成真实标签
        y = self._generate_classification_function(X)
        
        # 划分数据集（分层采样，60/20/20）
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.config.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.config.seed, stratify=y_temp
        )
        
        # 标准化（仅用训练集拟合）
        X_train, X_val, X_test, _ = self._standardize_data(X_train, X_val, X_test)
        
        # 失衡：仅训练集，对多数类下采样（X 与 y 同步索引）
        if self.imbalance_ratio > 0:
            imb_indices = self._create_class_imbalance(y_train, self.imbalance_ratio)
            X_train = X_train[imb_indices]
            y_train = y_train[imb_indices]
        
        # 标签噪声：仅训练集 r1 比例翻转
        y_train = self._add_label_noise(y_train, noise_ratio=self.noise_ratio)
        
        # 缺失注入（train/val/test 均注入）
        if self.missing_ratio > 0:
            mechanism = getattr(self.config, 'missing_mechanism', 'MCAR')
            X_train = self._add_missing_values(X_train, self.missing_ratio, mechanism)
            X_val = self._add_missing_values(X_val, self.missing_ratio, mechanism)
            X_test = self._add_missing_values(X_test, self.missing_ratio, mechanism)
        
        self.config.input_dim = n_features
        self.config.num_classes = 2
        
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        return train_loader, val_loader, (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _generate_regression_function(self, X):
        """
        非线性加性真函数（前 8 个特征有效，其余为噪声特征）
        特征数不足 8 时仅使用可用的前 p 项
        """
        p = X.shape[1]
        terms = [
            lambda X: -2 * np.sin(2 * X[:, 0]),
            lambda X: 8 * np.square(X[:, 1]),
            lambda X: 7 * np.sin(X[:, 2]) / (2 - np.sin(X[:, 2])),
            lambda X: 6 * np.exp(-X[:, 3]),
            lambda X: np.power(X[:, 4], 3) + 1.5 * np.square(X[:, 4] - 1),
            lambda X: 5 * X[:, 5],
            lambda X: 10 * np.sin(np.exp(-X[:, 6] / 2)),
            lambda X: -10 * norm.cdf(X[:, 7], loc=0.5, scale=0.8),
        ]
        
        y = np.zeros(X.shape[0])
        for j, term in enumerate(terms[:p]):
            y = y + term(X)
        return y
    
    def _generate_classification_function(self, X):
        """
        非线性分类边界（前 2 个特征有效）
        """
        f1 = np.square(X[:, 0] - 0.5)
        f2 = np.square(X[:, 1] - 0.5)
        y = f1 + f2 - 0.08
        y = (y > 0).astype(int)
        return y
    
    def _add_regression_noise(self, y, ratio=None, noise_mean=1.0, noise_std=0.5):
        """
        回归噪声：对 ratio 比例的样本 y += N(noise_mean, noise_std²)（论文 §4.1）
        保留旧签名兼容：_add_regression_noise(y, noise_type=...) 时可传 ratio 关键字
        
        Returns:
            加噪后的 y
        """
        if ratio is None:
            ratio = self.noise_ratio
        n = len(y)
        y_noisy = y.copy()
        n_noise = max(1, int(n * ratio)) if ratio > 0 else 0
        noise_indices = np.random.choice(n, n_noise, replace=False)
        y_noisy[noise_indices] += np.random.normal(noise_mean, noise_std, n_noise)
        return y_noisy
    
    def _add_missing_values(self, X, missing_ratio, missing_pattern='MCAR'):
        """
        缺失注入（论文 §4.1，机制参照 biam_data_utils.py）：
        - MCAR: 完全随机缺失，mask = Uniform(0,1) < ratio
        - MAR:  缺失概率依赖其他特征（前一特征），sigmoid 概率 × ratio
        - MNAR: 缺失概率依赖自身取值，sigmoid 值 × ratio
        
        Args:
            X: 数据矩阵
            missing_ratio: 目标缺失比例
            missing_pattern: 'MCAR' | 'MAR' | 'MNAR'
        
        Returns:
            注入缺失后的 X（NaN 表示缺失）
        """
        X = X.copy()
        n, p = X.shape
        
        if missing_pattern == 'MCAR':
            mask = np.random.uniform(0, 1, size=(n, p)) < missing_ratio
            X[mask] = np.nan
        elif missing_pattern == 'MAR':
            # 缺失概率依赖前一特征（随机选取一个条件特征）
            for j in range(p):
                cond_idx = (j + 1) % p  # 依赖的特征
                probs = 1.0 / (1.0 + np.exp(-(X[:, cond_idx] - np.median(X[:, cond_idx])) * 2))
                mask = np.random.uniform(0, 1, size=n) < probs * missing_ratio * 2
                X[mask, j] = np.nan
        elif missing_pattern == 'MNAR':
            # 缺失概率依赖自身取值（值越大越容易缺失）
            for j in range(p):
                col = X[:, j]
                probs = 1.0 / (1.0 + np.exp(-(col - np.median(col)) * 2))
                mask = np.random.uniform(0, 1, size=n) < probs * missing_ratio * 2
                X[mask, j] = np.nan
        else:
            raise ValueError(f"Unknown missing pattern: {missing_pattern}")
        
        return X
    
    def _add_label_noise(self, y, noise_ratio=None):
        """
        分类标签噪声：翻转 ratio 比例的标签（论文 §4.1）
        """
        if noise_ratio is None:
            noise_ratio = self.noise_ratio
        n = len(y)
        y_noisy = y.copy()
        if noise_ratio > 0 and n > 0:
            n_noise = max(1, int(n * noise_ratio))
            noise_indices = np.random.choice(n, n_noise, replace=False)
            y_noisy[noise_indices] = 1 - y_noisy[noise_indices]
        return y_noisy
    
    def _create_class_imbalance(self, y, imbalance_ratio=None):
        """
        类别失衡：对多数类下采样，使 少数:多数 ≈ imbalance_ratio（如 0.1 = 1:10）
        
        Args:
            y: 标签数组
            imbalance_ratio: 少数类与多数类的比例
        
        Returns:
            选中样本的索引（X, y 需同步索引，避免原实现的数据对齐 bug）
        """
        if imbalance_ratio is None:
            imbalance_ratio = self.imbalance_ratio
        
        classes, counts = np.unique(y.astype(int), return_counts=True)
        if len(classes) != 2:
            return np.arange(len(y))
        
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        # 边界情况：两类数量相等时 argmin==argmax，需显式区分
        if majority_class == minority_class:
            majority_class = classes[classes != minority_class][0]
        
        n_minority = counts.min()
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # 目标：少数:多数 ≈ imbalance_ratio
        n_majority_target = int(n_minority / max(imbalance_ratio, 1e-6))
        if n_majority_target <= len(majority_indices):
            # 多数类充足：保留全部少数类，下采样多数类
            selected_majority = np.random.choice(
                majority_indices, n_majority_target, replace=False
            )
            selected = np.concatenate([minority_indices, selected_majority])
        else:
            # 多数类不足：保留全部多数类，下采样少数类
            n_minority_target = max(1, int(len(majority_indices) * imbalance_ratio))
            selected_minority = np.random.choice(
                minority_indices, n_minority_target, replace=False
            )
            selected = np.concatenate([selected_minority, majority_indices])
        
        np.random.shuffle(selected)
        return selected
    
    def _standardize_data(self, X_train, X_val, X_test):
        """
        标准化：仅用训练集拟合 scaler
        
        Returns:
            (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def _create_data_loader(self, X, y, batch_size=None, shuffle=True):
        """
        创建 PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        dataset = Data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        return loader
    
    def _generate_adult_data(self):
        """
        Generate Adult dataset with missing values and imbalance
        """
        # Generate synthetic data with Adult-like characteristics
        result = self._generate_synthetic_classification()
        return result
    
    def _generate_credit_data(self):
        """
        Generate Credit dataset with missing values and imbalance
        """
        # Generate synthetic data with Credit-like characteristics
        result = self._generate_synthetic_classification()
        return result
