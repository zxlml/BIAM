"""
Performance benchmark tests for BIAM model
（轻量化：在小数据规模上验证性能与稳定性）
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch
import numpy as np
import time
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from models.biam_additive_model import BIAMAdditiveModel
from data.biam_data_generator import BIAMDataGenerator
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_config import BIAMConfig
from utils.biam_memory_optimizer import BIAMMemoryOptimizer
from utils.biam_batch_optimizer import BIAMBatchOptimizer

def make_perf_config(task='classification'):
    """性能测试配置（小规模）"""
    config = BIAMConfig()
    config.task = task
    config.input_dim = 20
    config.num_classes = 2
    config.batch_size = 128
    config.n_samples = 2000
    config.n_features = 20
    config.missing_ratio = 0.2
    config.noise_ratio = 0.2
    config.imbalance_ratio = 0.2
    config.device = torch.device('cpu')
    return config

class TestBIAMPerformance(unittest.TestCase):
    """
    Performance benchmark tests for BIAM model
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.config = make_perf_config('classification')
        self.n_features = 20
        self.n_samples = 2000
        
        torch.manual_seed(0)
        np.random.seed(0)
        self.X = torch.randn(self.n_samples, self.n_features)
        self.y = torch.randint(0, 2, (self.n_samples,))
        
        self.data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X, self.y),
            batch_size=self.config.batch_size,
            shuffle=True
        )
    
    def test_model_initialization_performance(self):
        """
        Test model initialization performance
        """
        start_time = time.time()
        biam_model = BIAMModel(self.config, self.config.device)
        init_time = time.time() - start_time
        
        self.assertLess(init_time, 5.0)
        print(f"Model initialization time: {init_time:.4f}s")
    
    def test_forward_pass_performance(self):
        """
        Test forward pass performance
        """
        biam_model = BIAMModel(self.config, self.config.device)
        biam_model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = biam_model(self.X[:100])
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = biam_model(self.X[:100])
        forward_time = (time.time() - start_time) / 10
        
        self.assertLess(forward_time, 1.0)
        print(f"Forward pass time per batch: {forward_time:.4f}s")
    
    def test_training_performance(self):
        """
        Test training performance
        """
        biam_model = BIAMModel(self.config, self.config.device)
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        optimizer = BIAMOptimizer(self.config, biam_model, weighting_network)
        
        start_time = time.time()
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X[:256], self.y[:256]),
            batch_size=128
        )
        for i in range(2):
            train_metrics = optimizer.train_epoch(self.data_loader, val_loader, i)
        
        training_time = time.time() - start_time
        
        self.assertLess(training_time, 60.0)
        print(f"Training time for 2 epochs: {training_time:.4f}s")
    
    def test_memory_usage(self):
        """
        Test memory usage
        """
        memory_optimizer = BIAMMemoryOptimizer(self.config.device)
        
        initial_memory = memory_optimizer.get_memory_usage()
        biam_model = BIAMModel(self.config, self.config.device)
        model_memory = memory_optimizer.get_memory_usage()
        
        memory_increase = model_memory.get('gpu_allocated', 0) - initial_memory.get('gpu_allocated', 0)
        
        self.assertLess(memory_increase, 1.0)
        print(f"Memory increase: {memory_increase:.3f} GB")
    
    def test_batch_processing_performance(self):
        """
        Test batch processing performance
        """
        batch_optimizer = BIAMBatchOptimizer(self.config)
        
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        
        start_time = time.time()
        dataloader = batch_optimizer.create_optimized_dataloader(
            dataset, batch_size=self.config.batch_size
        )
        
        for i, (data, target) in enumerate(dataloader):
            if i >= 5:
                break
        
        loading_time = time.time() - start_time
        
        self.assertLess(loading_time, 10.0)
        print(f"Data loading time for 5 batches: {loading_time:.4f}s")
    
    def test_gradient_computation_performance(self):
        """
        Test gradient computation performance
        """
        biam_model = BIAMModel(self.config, self.config.device)
        biam_model.train()
        
        start_time = time.time()
        
        for i, (data, target) in enumerate(self.data_loader):
            if i >= 3:
                break
            
            output = biam_model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            
            loss.backward()
            biam_model.zero_grad()
        
        gradient_time = time.time() - start_time
        
        self.assertLess(gradient_time, 20.0)
        print(f"Gradient computation time for 3 batches: {gradient_time:.4f}s")
    
    def test_model_compression_performance(self):
        """
        Test model compression performance
        """
        from utils.biam_model_compression import BIAMModelCompression
        
        compression = BIAMModelCompression(self.config)
        biam_model = BIAMModel(self.config, self.config.device)
        
        start_time = time.time()
        compressed_model = compression.prune_model(biam_model, pruning_ratio=0.2)
        compression_time = time.time() - start_time
        
        self.assertLess(compression_time, 10.0)
        print(f"Model compression time: {compression_time:.4f}s")
        
        compressed_model.eval()
        with torch.no_grad():
            _ = compressed_model(self.X[:100])
    
    def test_data_generation_performance(self):
        """
        Test data generation performance
        """
        generator = BIAMDataGenerator(self.config)
        
        start_time = time.time()
        result = generator.generate_data()
        generation_time = time.time() - start_time
        
        self.assertLess(generation_time, 15.0)
        print(f"Data generation time: {generation_time:.4f}s")
    
    def test_end_to_end_performance(self):
        """
        Test end-to-end performance
        """
        generator = BIAMDataGenerator(self.config)
        train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()
        
        biam_model = BIAMModel(self.config, self.config.device)
        # 用训练数据设置 hinge 节点
        biam_model.additive_model.set_knots(torch.tensor(train_data[0], dtype=torch.float32))
        
        weighting_network = BIAMWeightingNetwork(self.config, self.config.device)
        optimizer = BIAMOptimizer(self.config, biam_model, weighting_network)
        
        start_time = time.time()
        
        for epoch in range(2):
            train_metrics = optimizer.train_epoch(train_loader, val_loader, epoch)
            if epoch % 2 == 0:
                test_metrics = optimizer.evaluate(test_data)
        
        end_to_end_time = time.time() - start_time
        
        self.assertLess(end_to_end_time, 120.0)
        print(f"End-to-end training time for 2 epochs: {end_to_end_time:.4f}s")
    
    def test_scalability(self):
        """
        Test model scalability with different data sizes
        """
        sizes = [100, 500, 1000, 2000]
        times = []
        
        biam_model = BIAMModel(self.config, self.config.device)
        biam_model.eval()
        
        for size in sizes:
            X = torch.randn(size, self.n_features)

            # 重复多次取最小值，消除 CPU 计时抖动
            forward_time = min(
                self._timed_forward(biam_model, X) for _ in range(5)
            )
            times.append(forward_time)

            print(f"Size {size}: {forward_time:.4f}s")

        # 时间应随数据规模大致线性（允许开销）
        for i in range(1, len(times)):
            time_ratio = times[i] / max(times[i-1], 1e-6)
            size_ratio = sizes[i] / sizes[i-1]
            self.assertLessEqual(time_ratio, size_ratio * 2.0)

    @staticmethod
    def _timed_forward(biam_model, X):
        start_time = time.time()
        with torch.no_grad():
            _ = biam_model(X)
        return time.time() - start_time

if __name__ == '__main__':
    unittest.main()
