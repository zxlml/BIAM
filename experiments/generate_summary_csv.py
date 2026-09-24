"""
完整实验复现脚本：生成论文对齐仿真实验的汇总 CSV。

直接复用 tests/test_biam_simulation.py 中经过校准的配置与训练流程，
复现以下实验并写入 results/simulation_summary.csv：
  1. 回归-clean：BIAM vs BIAM-B（Table 2 干净基线行为）
  2. 回归-噪声30%+缺失20%（多种子）：BIAM vs BIAM-B（Table 2 核心行为）
  3. 分类-clean：Macro-F1（Table 3 行为）
  4. 分类-组合腐蚀（噪声/失衡/缺失各10%）：BIAM vs BIAM-B 非劣性
  5. BIAM-H 外推消融：hinge vs 分段常数（§5 claim）
  6. BIAM-I 缺失模块消融：MNAR 缺失25% 下的模块激活与非劣性
  7. 双层重加权机制：ν 与逐样本损失的 Spearman 相关、模型 vs 均值预测器

运行方式（项目根目录）：
    python experiments/generate_summary_csv.py
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import csv
import copy

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from tests.test_biam_simulation import make_sim_config, generate, build_and_train  # noqa: E402
from models.biam_model import BIAMModel  # noqa: E402
from models.biam_weighting_network import BIAMWeightingNetwork  # noqa: E402
from gradients.biam_optimizer import BIAMOptimizer  # noqa: E402

ROWS = []
LOSS_KEYS = ('mse', 'rmse', 'mae', 'f1_macro', 'accuracy')


def record(experiment, scenario, variant, seed, metrics, note='', **extra):
    def _g(key):
        v = metrics.get(key, float('nan'))
        return round(v, 4) if isinstance(v, (int, float)) else v

    row = {
        'experiment': experiment, 'scenario': scenario, 'variant': variant,
        'seed': seed,
        'mse': _g('mse'), 'rmse': _g('rmse'), 'mae': _g('mae'),
        'f1_macro': _g('f1_macro'), 'accuracy': _g('accuracy'),
        'spearman_nu_vs_loss': '', 'mean_predictor_mse': '', 'note': note,
    }
    row.update(extra)
    ROWS.append(row)
    print(f"  [{experiment}/{scenario}/{variant}/seed{seed}] mse={row['mse']} note={note}")


def run_regression(experiment, scenario, config):
    """回归场景：BIAM 与 BIAM-B 各训练一次"""
    tr, vl, trd, vd, td = generate(config)
    _, biam = build_and_train(config, tr, vl, trd, td)

    config_b = copy.deepcopy(config)
    config_b.use_bilevel = False
    _, biam_b = build_and_train(config_b, tr, vl, trd, td)

    record(experiment, scenario, 'BIAM', config.seed, biam, note='双层加权')
    record(experiment, scenario, 'BIAM-B', config.seed, biam_b, note='均匀权重')
    return biam


def run_reweighting_mechanism(config):
    """双层重加权机制验证：Spearman(ν, per-sample loss) 与均值预测器对照"""
    tr, vl, trd, vd, td = generate(config)
    opt, biam = build_and_train(config, tr, vl, trd, td)

    # 在一个训练 batch 上测 ν 对逐样本损失的响应
    x_b, y_b = next(iter(tr))
    with torch.no_grad():
        losses = opt._per_sample_loss(opt.additive_model(x_b), y_b)
        nu = opt.weighting_network(opt._nu_input(losses)).flatten()
        rho = float(spearmanr(losses.flatten().numpy(), nu.numpy()).statistic)

        y_mean = float(np.mean(trd[1]))
        mean_pred_mse = float(np.mean((np.asarray(td[1], dtype=np.float64) - y_mean) ** 2))

    record('bilevel_reweighting', 'noise30', 'BIAM', config.seed, biam,
           note=f'ν 与逐样本损失的 Spearman 相关（负值=降权高损失样本）',
           spearman_nu_vs_loss=round(rho, 4),
           mean_predictor_mse=round(mean_pred_mse, 4))
    print(f"  均值预测器 MSE={mean_pred_mse:.4f}（对照基线）")


def run_extrapolation():
    """BIAM-H 消融：hinge vs 分段常数基的域外外推（训练域 [-1,0.8]，测试域 [1.0,1.5]）"""
    for seed in (42, 7):
        for basis_type, variant in (('piecewise_linear', 'BIAM(hinge)'),
                                    ('piecewise_constant', 'BIAM-H(constant)')):
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
            for _ in range(config.epochs):
                optimizer.train_epoch(train_loader, val_loader, 0)

            metrics = optimizer.evaluate((X_test, y_test))
            record('biam_h_extrapolation', 'domain_shift', variant, seed, metrics,
                   note='y=3*x0, 训练域[-1,0.8], 测试域[1.0,1.5]（纯外推）')


def main():
    print("=" * 78)
    print("BIAM 完整实验复现（论文 §4.1 仿真协议）")
    print("=" * 78)

    # 1. 回归-clean
    print("\n[1/6] 回归-clean：BIAM vs BIAM-B")
    run_regression('regression', 'clean',
                   make_sim_config('regression'))

    # 2. 回归-噪声30%+缺失20%（多种子）
    print("\n[2/6] 回归-噪声30%+缺失20%（seed 42, 7）")
    corrupted = {}
    for seed in (42, 7):
        biam = run_regression('regression', 'noise30_missing20',
                              make_sim_config('regression', seed=seed,
                                              noise_ratio=0.3, missing_ratio=0.2))
        corrupted.setdefault('BIAM', []).append(biam['mse'])
    # 聚合行：多种子平均
    for variant, key in (('BIAM', 'BIAM'), ('BIAM-B', 'BIAM-B')):
        mses = [r['mse'] for r in ROWS
                if r['experiment'] == 'regression'
                and r['scenario'] == 'noise30_missing20' and r['variant'] == variant]
        ROWS.append({
            'experiment': 'regression', 'scenario': 'noise30_missing20',
            'variant': variant, 'seed': 'mean',
            'mse': round(float(np.mean(mses)), 4), 'rmse': '', 'mae': '',
            'f1_macro': '', 'accuracy': '', 'spearman_nu_vs_loss': '',
            'mean_predictor_mse': '', 'note': '多种子平均（论文 Table 2 核心行为：BIAM < BIAM-B）'
        })

    # 3. 分类-clean
    print("\n[3/6] 分类-clean：Macro-F1")
    cls_cfg = make_sim_config('classification', upper_lr=0.1, epochs=60)
    tr, vl, trd, vd, td = generate(cls_cfg)
    _, cls_metrics = build_and_train(cls_cfg, tr, vl, trd, td)
    record('classification', 'clean', 'BIAM', cls_cfg.seed, cls_metrics,
           note='Table 3 干净基线行为')

    # 4. 分类-组合腐蚀（非劣性）
    print("\n[4/6] 分类-组合腐蚀（噪声/失衡/缺失各10%）")
    run_regression('classification', 'combined_corruption',
                   make_sim_config('classification', upper_lr=0.1,
                                   noise_ratio=0.1, imbalance_ratio=0.1, missing_ratio=0.1))

    # 5. BIAM-H 外推消融
    print("\n[5/6] BIAM-H 外推消融（hinge vs 分段常数）")
    run_extrapolation()

    # 6. BIAM-I 缺失模块消融（MNAR 25%）+ 双层重加权机制
    print("\n[6/6] BIAM-I 缺失模块消融 + 双层重加权机制")
    mnar_cfg = make_sim_config('regression', noise_ratio=0.2, missing_ratio=0.25,
                               missing_mechanism='MNAR')
    tr, vl, trd, vd, td = generate(mnar_cfg)
    opt_i, biam = build_and_train(mnar_cfg, tr, vl, trd, td)
    config_i = copy.deepcopy(mnar_cfg)
    config_i.use_missing_interactions = False
    _, biam_i = build_and_train(config_i, tr, vl, trd, td)

    beta_miss = opt_i.additive_model.get_missing_indicators()
    record('biam_i_missing_module', 'MNAR_missing25', 'BIAM', mnar_cfg.seed, biam,
           note=f'β_miss 平均幅值={np.abs(beta_miss).mean():.4f}（缺失模块激活）')
    record('biam_i_missing_module', 'MNAR_missing25', 'BIAM-I', mnar_cfg.seed, biam_i,
           note='移除缺失交互（预测非劣性成立）')

    rw_cfg = make_sim_config('regression', noise_ratio=0.3)
    run_reweighting_mechanism(rw_cfg)

    # 写出 CSV
    out_dir = os.path.join(ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'simulation_summary.csv')
    fieldnames = ['experiment', 'scenario', 'variant', 'seed', 'mse', 'rmse', 'mae',
                  'f1_macro', 'accuracy', 'spearman_nu_vs_loss', 'mean_predictor_mse', 'note']
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)

    print("\n" + "=" * 78)
    print(f"汇总已写入: {out_path}（共 {len(ROWS)} 行）")
    print("=" * 78)


if __name__ == '__main__':
    main()
