"""
BIAM Configuration
Configuration class for BIAM model parameters
"""

import torch
from typing import Dict, Any, Optional

class BIAMConfig:
    """
    Configuration class for BIAM model
    """
    
    def __init__(self, args=None):
        """
        Initialize BIAM configuration
        
        Args:
            args: Command line arguments or configuration dictionary
        """
        # Default configuration
        self.task = 'classification'
        self.dataset = 'synthetic'
        self.missing_ratio = 0.3
        self.noise_ratio = 0.2
        self.imbalance_ratio = 0.15
        self.upper_lr = 1e-2
        self.lower_lr = 1e-2
        self.penalty_coef = 1e-5
        self.epochs = 200
        self.batch_size = 200
        self.use_wandb = False
        self.project_name = 'biam-experiments'
        
        # 随机种子（保证仿真可复现）
        self.seed = 42
        
        # 数据规模参数（论文 §4.1）
        self.n_samples = 1200
        self.n_features = 10
        
        # 回归标签噪声参数：ε ~ N(μe, σe)（论文 Eq.16 相关设定）
        self.noise_mean = 1.0   # μe
        self.noise_std = 0.5    # σe
        
        # 缺失机制：MCAR / MAR / MNAR（论文 §4.1）
        self.missing_mechanism = 'MCAR'
        
        # 加性模型 hinge 基节点数 L（节点取训练数据分位数）
        self.n_knots = 8
        
        # 正则系数：λ1(ℓ2) 与 λ2(ℓ0)，对应论文 Eq.2 中的 λ1‖w‖₂² + λ2‖w‖₀
        self.lambda_l2 = 1e-3
        self.lambda_l0 = 1e-4
        
        # 基函数类型：'piecewise_linear'（hinge, 论文默认）或 'piecewise_constant'（BIAM-H 消融）
        self.basis_type = 'piecewise_linear'
        
        # 是否启用缺失交互项 Σ α_{j,k,τ} I(m_j=1) h(x_k;η_τ)（BIAM-I 消融时置 False）
        self.use_missing_interactions = True
        
        # 是否启用双层优化（BIAM-B 消融时置 False，退化为固定均匀样本权重）
        self.use_bilevel = True
        
        # Model architecture parameters
        self.input_dim = 100
        self.num_classes = 2
        self.spline_dim_regression = 3
        self.spline_dim_classification = 5
        self.hidden_dim_weighting = 10
        
        # Optimization parameters
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.scheduler_step_size = 1000
        self.scheduler_gamma = 0.1
        
        # Regularization parameters
        self.regularization_type = 'group_lasso'
        self.dropout_rate = 0.1
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Logging parameters
        self.log_interval = 100
        self.save_interval = 1000
        self.eval_interval = 100
        
        # Data parameters
        self.num_workers = 0
        self.pin_memory = True
        self.shuffle_train = True
        self.shuffle_val = False
        
        # Update with provided arguments
        if args is not None:
            self._update_from_args(args)
    
    def _update_from_args(self, args):
        """
        Update configuration from command line arguments
        
        Args:
            args: Command line arguments
        """
        # 通用更新：只要 args 中存在同名属性且 config 已定义，就同步
        for key, value in vars(args).items():
            if hasattr(self, key) and key not in ('device',):
                setattr(self, key, value)
    
    def get_spline_dim(self):
        """
        Get spline dimension based on task
        
        Returns:
            Spline dimension
        """
        if self.task == 'regression':
            return self.spline_dim_regression
        else:
            return self.spline_dim_classification
    
    def get_output_dim(self):
        """
        Get output dimension based on task
        
        Returns:
            Output dimension
        """
        if self.task == 'regression':
            return 1
        else:
            return self.num_classes
    
    def to_dict(self):
        """
        Convert configuration to dictionary
        
        Returns:
            Configuration dictionary
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        return config_dict
    
    def update(self, **kwargs):
        """
        Update configuration parameters
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def validate(self):
        """
        Validate configuration parameters
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.task not in ['regression', 'classification']:
            raise ValueError(f"Invalid task: {self.task}")
        
        if self.upper_lr <= 0 or self.lower_lr <= 0:
            raise ValueError("Learning rates must be positive")
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if not 0 <= self.missing_ratio <= 1:
            raise ValueError("Missing ratio must be between 0 and 1")
        
        if not 0 <= self.noise_ratio <= 1:
            raise ValueError("Noise ratio must be between 0 and 1")
        
        if not 0 <= self.imbalance_ratio <= 1:
            raise ValueError("Imbalance ratio must be between 0 and 1")
        
        if self.missing_mechanism not in ['MCAR', 'MAR', 'MNAR']:
            raise ValueError(f"Invalid missing mechanism: {self.missing_mechanism}")
        
        if self.basis_type not in ['piecewise_linear', 'piecewise_constant']:
            raise ValueError(f"Invalid basis type: {self.basis_type}")
    
    def __str__(self):
        """
        String representation of configuration
        
        Returns:
            Configuration string
        """
        config_str = "BIAM Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str
    
    def __repr__(self):
        """
        Representation of configuration
        
        Returns:
            Configuration representation
        """
        return f"BIAMConfig({self.to_dict()})"
