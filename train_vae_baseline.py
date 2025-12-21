"""
训练VAE Baseline模型用于与Diffusion模型对比

使用说明:
    python train_vae_baseline.py --device cuda:0

内存优化方案:
    - 使用较小batch_size (256/512) 而非2048
    - 可选: gradient accumulation模拟大batch效果
"""

import argparse
import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add moses to path
sys.path.insert(0, str(Path(__file__).parent))

import moses
from moses.vae import VAE, VAETrainer
from moses.vae.config import get_parser


def load_moses2_data(csv_path, split='train'):
    """
    从moses2.csv加载指定split的数据

    Args:
        csv_path: moses2.csv路径
        split: 'train', 'test', or 'test_scaffolds'

    Returns:
        list of SMILES strings
    """
    print(f"Loading {split} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if split == 'test_scaffolds':
        split_name = 'test_scaffolds'
    else:
        split_name = split

    data = df[df['SPLIT'] == split_name]['SMILES'].tolist()
    print(f"Loaded {len(data)} molecules for {split}")
    return data


def get_comparison_config(args):
    """
    获取与diffusion模型对齐的VAE配置
    """
    # 使用MOSES默认parser
    parser = get_parser()

    # 解析参数
    if hasattr(args, '__dict__'):
        config_args = []
        for key, value in vars(args).items():
            if value is not None and key not in ['device', 'use_moses2', 'moses2_path']:
                config_args.extend([f'--{key}', str(value)])
        config = parser.parse_args(config_args)
    else:
        config = parser.parse_args([])

    # ========== 对齐diffusion模型的参数 ==========

    # 模型架构
    config.q_bidir = True              # 双向GRU
    config.q_d_h = 256                 # 编码器隐藏层
    config.q_n_layers = 1
    config.q_dropout = 0.1             # ✅ 与diffusion对齐 (降低自0.5)

    config.d_z = 128                   # ✅ 潜在维度 = diffusion hidden_dim
    config.d_d_h = 512                 # 解码器隐藏层
    config.d_n_layers = 3
    config.d_dropout = 0.1             # ✅ 与diffusion对齐

    # 训练参数
    config.n_batch = args.n_batch if hasattr(args, 'n_batch') else 512
    config.lr_start = 1e-4             # ✅ 与diffusion对齐
    config.lr_end = 1e-4
    config.clip_grad = 50

    # KL权重
    config.kl_start = 0
    config.kl_w_start = 0.0
    config.kl_w_end = 0.05             # 可调: 0.01-0.1

    # Learning rate schedule (SGDR)
    # 要训练约40 epochs达到30k steps
    # SGDR总epoch = sum(period * mult^i for i in range(restarts))
    # 例: period=5, restarts=4, mult=2 -> 5+10+20+40=75 epochs (太多)
    # 改为: period=10, restarts=2, mult=2 -> 10+20=30 epochs
    config.lr_n_period = 10
    config.lr_n_restarts = 4           # 10+20+40+80=150 epochs (如需40 epochs用2)
    config.lr_n_mult = 1               # 不增长周期，保持10 epochs per restart

    # 其他
    config.n_workers = 4
    config.n_jobs = 1
    config.n_last = 1000

    # 保存配置
    config.save_frequency = 5          # 每5个epoch保存一次

    return config


def main():
    parser = argparse.ArgumentParser(description='Train VAE Baseline for Comparison with Diffusion')

    # 数据选项
    parser.add_argument('--use_moses2', action='store_true',
                        help='Use moses2.csv instead of MOSES official dataset')
    parser.add_argument('--moses2_path', type=str, default='./datasets/moses2.csv',
                        help='Path to moses2.csv')

    # 训练参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--n_batch', type=int, default=512,
                        help='Batch size (512 or 256 for memory constraint, original diffusion=2048)')
    parser.add_argument('--seed', type=int, default=102,
                        help='Random seed (use 102 to align with diffusion)')

    # 模型保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints/vae_baseline',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='./checkpoints/vae_baseline/log.txt',
                        help='Log file path')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 获取配置
    config = get_comparison_config(args)
    config.log_file = args.log_file
    config.model_save = os.path.join(args.save_dir, 'model.pt')
    config.config_save = os.path.join(args.save_dir, 'config.pt')
    config.vocab_save = os.path.join(args.save_dir, 'vocab.pt')

    # 加载数据
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)

    if args.use_moses2:
        train_data = load_moses2_data(args.moses2_path, 'train')
        test_data = load_moses2_data(args.moses2_path, 'test')
    else:
        print("Loading MOSES official dataset...")
        train_data = moses.get_dataset('train')
        test_data = moses.get_dataset('test')
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # 打印配置对比
    print("\n" + "="*60)
    print("VAE Configuration (Aligned with Diffusion)")
    print("="*60)
    print(f"{'Parameter':<25} {'Value':<20} {'Note':<20}")
    print("-"*65)
    print(f"{'Latent Dim (d_z)':<25} {config.d_z:<20} {'✅ = diffusion 128':<20}")
    print(f"{'Batch Size':<25} {config.n_batch:<20} {'⚠️ diffusion=2048':<20}")
    print(f"{'Learning Rate':<25} {config.lr_start:<20} {'✅ = diffusion 1e-4':<20}")
    print(f"{'Dropout':<25} {config.q_dropout:<20} {'✅ = diffusion 0.1':<20}")
    print(f"{'Seed':<25} {args.seed:<20} {'✅ = diffusion 102':<20}")

    # 计算训练步数
    steps_per_epoch = len(train_data) // config.n_batch
    total_epochs = config.lr_n_period * config.lr_n_restarts
    total_steps = steps_per_epoch * total_epochs
    target_steps = 30000

    print(f"{'Steps per Epoch':<25} {steps_per_epoch:<20}")
    print(f"{'Total Epochs':<25} {total_epochs:<20}")
    print(f"{'Total Steps':<25} {total_steps:<20} {f'Target: {target_steps}':<20}")
    print("-"*65)

    if total_steps < target_steps * 0.8:
        print(f"⚠️  Warning: Total steps ({total_steps}) < target ({target_steps})")
        print(f"   Consider increasing lr_n_restarts to {int(target_steps/steps_per_epoch/config.lr_n_period) + 1}")

    # 创建模型
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)

    trainer = VAETrainer(config)
    vocab = trainer.get_vocabulary(train_data)
    model = VAE(vocab, config).to(args.device)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 保存配置和词汇表
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)
    print(f"\n✅ Config saved to {config.config_save}")
    print(f"✅ Vocab saved to {config.vocab_save}")

    # 训练
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Batch size: {config.n_batch}")
    print(f"Total epochs: {total_epochs}")
    print("="*60 + "\n")

    try:
        model = trainer.fit(model, train_data, val_data=test_data)

        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, 'model_final.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"\n✅ Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel checkpoints saved in: {args.save_dir}")
    print(f"Log file: {args.log_file}")
    print("\nNext steps:")
    print("1. Generate samples: python scripts/sample.py vae --model_load ...")
    print("2. Evaluate metrics: python scripts/eval.py --gen_path ... --ref_path ...")


if __name__ == '__main__':
    main()
