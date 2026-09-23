"""
BIAM: Bilevel Interactive Additive Model
Main entry point for training and evaluation
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# wandb 为可选依赖
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_logger import BIAMLogger
from utils.biam_config import BIAMConfig

def main():
    """
    Main function for BIAM training and evaluation
    """
    parser = argparse.ArgumentParser(description='BIAM: Bilevel Interactive Additive Model')
    parser.add_argument('--task', type=str, default='classification', 
                       choices=['regression', 'classification'],
                       help='Task type: regression or classification')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'adult', 'credit', 'mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--missing_ratio', type=float, default=0.3,
                       help='Ratio of missing values')
    parser.add_argument('--noise_ratio', type=float, default=0.2,
                       help='Ratio of noisy labels')
    parser.add_argument('--imbalance_ratio', type=float, default=0.15,
                       help='Ratio for class imbalance (minority:majority)')
    parser.add_argument('--noise_mean', type=float, default=1.0,
                       help='回归噪声均值 μe（噪声 ε ~ N(μe, σe)）')
    parser.add_argument('--noise_std', type=float, default=0.5,
                       help='回归噪声标准差 σe')
    parser.add_argument('--missing_mechanism', type=str, default='MCAR',
                       choices=['MCAR', 'MAR', 'MNAR'],
                       help='缺失机制')
    parser.add_argument('--n_samples', type=int, default=1200,
                       help='仿真样本数')
    parser.add_argument('--n_features', type=int, default=10,
                       help='仿真特征数')
    parser.add_argument('--n_knots', type=int, default=8,
                       help='hinge 基节点数 L')
    parser.add_argument('--lambda_l2', type=float, default=1e-3,
                       help='ℓ2 正则系数 λ1')
    parser.add_argument('--lambda_l0', type=float, default=1e-4,
                       help='ℓ0 正则系数 λ2')
    parser.add_argument('--basis_type', type=str, default='piecewise_linear',
                       choices=['piecewise_linear', 'piecewise_constant'],
                       help='基函数类型（piecewise_constant 用于 BIAM-H 消融）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--upper_lr', type=float, default=1e-2,
                       help='Upper level learning rate')
    parser.add_argument('--lower_lr', type=float, default=1e-2,
                       help='Lower level learning rate')
    parser.add_argument('--penalty_coef', type=float, default=1e-5,
                       help='Penalty coefficient for regularization')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch size')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--project_name', type=str, default='biam-experiments',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = BIAMConfig(args)
    config.validate()
    
    # Initialize logger
    logger = BIAMLogger(config)
    
    # Initialize wandb if requested
    if args.use_wandb:
        if not HAS_WANDB:
            logger.info("wandb 未安装，跳过 wandb 日志记录")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.project_name,
                config=vars(args),
                name=f"biam_{args.task}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate data
    data_generator = BIAMDataGenerator(config)
    train_loader, val_loader, train_data, val_data, test_data = data_generator.generate_data()
    
    # Initialize BIAM model
    biam_model = BIAMModel(config, device)
    weighting_network = BIAMWeightingNetwork(config, device)
    
    # 用训练数据分位数设置 hinge 节点 η_τ（论文：节点预定义为训练数据分位数）
    X_train_tensor = torch.tensor(train_data[0], dtype=torch.float32)
    biam_model.additive_model.set_knots(X_train_tensor)
    
    # Initialize optimizer
    optimizer = BIAMOptimizer(config, biam_model, weighting_network)
    
    # Training loop
    logger.info("Starting BIAM training...")
    for epoch in range(args.epochs):
        # Train one epoch
        train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
        
        # Evaluate on test set
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            test_metrics = optimizer.evaluate(test_data)
            
            # Log metrics
            logger.log_metrics(epoch, train_metrics, test_metrics)
            
            if args.use_wandb and HAS_WANDB:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': train_metrics['val_loss'],
                }
                if config.task == 'classification':
                    log_dict.update({
                        'test_accuracy': test_metrics['accuracy'],
                        'test_f1_macro': test_metrics['f1_macro']
                    })
                else:
                    log_dict.update({'test_mse': test_metrics['mse']})
                wandb.log(log_dict)
    
    # 最终评估
    final_metrics = optimizer.evaluate(test_data)
    logger.info(f"Final test metrics: {final_metrics}")
    logger.info("Training completed!")
    
    if args.use_wandb and HAS_WANDB:
        wandb.finish()

if __name__ == '__main__':
    main()
