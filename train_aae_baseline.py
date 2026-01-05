"""
训练AAE (Adversarial Autoencoder) Baseline模型用于与Diffusion模型对比

AAE使用对抗训练来约束潜在空间分布，相比VAE的KL散度约束，
AAE使用判别器来强制潜在编码符合先验分布（通常是高斯分布）。

使用说明:
    python train_aae_baseline.py --device cuda:0
    或使用CPU（如果GPU有问题）:
    python train_aae_baseline.py --device cpu
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
from moses.aae import AAE, AAETrainer
from moses.aae.config import get_parser


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


def get_comparison_config(n_batch=512, lr=1e-3):
    """
    获取AAE的推荐配置（平衡MOSES标准和对比实验需求）

    AAE关键参数:
    - latent_size: 128 (与diffusion hidden_dim对齐)
    - learning_rate: 1e-3 (AAE标准，比diffusion高)
    - encoder_hidden_size: 512 (MOSES标准)
    - train_epochs: 50 (折中，原MOSES=120，但为了与其他baseline对齐)
    - batch_size: 512 (折中值，diffusion使用2048)
    """
    # 使用MOSES默认parser创建基础配置
    parser = get_parser()
    config = parser.parse_args([])

    # ========== AAE参数设置 ==========

    # 模型架构
    config.embedding_size = 32             # 嵌入维度 (MOSES标准)
    config.encoder_hidden_size = 512       # ✅ 编码器隐藏层 (恢复MOSES标准)
    config.encoder_num_layers = 1          # 编码器层数
    config.encoder_bidirectional = True    # 双向LSTM
    config.encoder_dropout = 0             # ✅ 恢复MOSES默认(AAE对dropout敏感)

    config.latent_size = 128               # ✅ 潜在维度 = diffusion hidden_dim
    config.decoder_hidden_size = 512       # 解码器隐藏层
    config.decoder_num_layers = 2          # 解码器层数
    config.decoder_dropout = 0             # ✅ 恢复MOSES默认

    config.discriminator_layers = [640, 256]  # 判别器层

    # 训练参数
    config.pretrain_epochs = 10            # ⚠️ 重要：预训练帮助稳定AAE
    config.train_epochs = 50               # ⚠️ 增加训练轮数(原120，折中为50)
    config.n_batch = n_batch               # batch size
    config.lr = lr                         # ✅ 学习率 = AAE标准 1e-3

    # 学习率调度
    config.step_size = 20                  # 每20个epoch衰减
    config.gamma = 0.5                     # 衰减因子0.5

    # 对抗训练
    config.discriminator_steps = 1         # 每个自编码器步骤训练判别器1次
    config.weight_decay = 0                # 权重衰减

    # 其他
    config.n_workers = 4
    config.n_jobs = 1
    config.save_frequency = 10             # 每10个epoch保存一次

    return config


def main():
    parser = argparse.ArgumentParser(description='Train AAE Baseline for Comparison with Diffusion')

    # 数据选项
    parser.add_argument('--use_moses2', action='store_true',
                        help='Use moses2.csv instead of MOSES official dataset')
    parser.add_argument('--moses2_path', type=str, default='./data/moses2.csv',
                        help='Path to moses2.csv')

    # 训练参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--n_batch', type=int, default=512,
                        help='Batch size (256/512/1024, original AAE=512)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (AAE standard=1e-3, diffusion=1e-4)')
    parser.add_argument('--seed', type=int, default=102,
                        help='Random seed (use 102 to align with diffusion)')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='Pretraining epochs (helps stabilize AAE)')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='Training epochs (original AAE=120, reduced for efficiency)')

    # 模型保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints/aae_baseline',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='./checkpoints/aae_baseline/log.txt',
                        help='Log file path')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 设置cuDNN配置
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN benchmark mode enabled")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 获取配置
    config = get_comparison_config(n_batch=args.n_batch, lr=args.lr)
    # 允许命令行覆盖默认值
    config.pretrain_epochs = args.pretrain_epochs
    config.train_epochs = args.train_epochs
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
    print("\n" + "="*70)
    print("AAE Configuration (Optimized for AAE Training)")
    print("="*70)
    print(f"{'Parameter':<30} {'Value':<20} {'Note':<20}")
    print("-"*70)
    print(f"{'Model Type':<30} {'AAE':<20} {'Adversarial AE':<20}")
    print(f"{'Latent Size':<30} {config.latent_size:<20} {'✅ = diffusion 128':<20}")
    print(f"{'Encoder Hidden':<30} {config.encoder_hidden_size:<20} {'✅ MOSES std 512':<20}")
    print(f"{'Decoder Hidden':<30} {config.decoder_hidden_size:<20} {'✅ MOSES std 512':<20}")
    print(f"{'Batch Size':<30} {config.n_batch:<20} {'diffusion=2048':<20}")
    print(f"{'Learning Rate':<30} {config.lr:<20} {'✅ AAE std 1e-3':<20}")
    print(f"{'Encoder Dropout':<30} {config.encoder_dropout:<20} {'MOSES default':<20}")
    print(f"{'Decoder Dropout':<30} {config.decoder_dropout:<20} {'MOSES default':<20}")
    print(f"{'Pretrain Epochs':<30} {config.pretrain_epochs:<20} {'⚠️ Important!':<20}")
    print(f"{'Train Epochs':<30} {config.train_epochs:<20} {'MOSES=120':<20}")
    print(f"{'Seed':<30} {args.seed:<20} {'✅ = diffusion 102':<20}")

    # 计算训练步数
    steps_per_epoch = len(train_data) // config.n_batch
    pretrain_steps = steps_per_epoch * config.pretrain_epochs
    train_steps = steps_per_epoch * config.train_epochs
    total_steps = pretrain_steps + train_steps

    print(f"{'Steps per Epoch':<30} {steps_per_epoch:<20}")
    print(f"{'Pretrain Steps':<30} {pretrain_steps:<20} {f'{config.pretrain_epochs} epochs':<20}")
    print(f"{'Train Steps':<30} {train_steps:<20} {f'{config.train_epochs} epochs':<20}")
    print(f"{'Total Steps':<30} {total_steps:<20}")
    print("-"*70)

    print("\n⚠️  AAE Training Notes:")
    print("  - AAE uses adversarial training, which is more unstable than VAE")
    print("  - Pretraining helps stabilize the autoencoder before adversarial training")
    print("  - Higher learning rate (1e-3) is necessary for AAE convergence")
    print("  - Expect training to take longer than VAE/CharRNN")
    print("  - Validity should reach >90% with proper training")

    # 创建模型
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)

    trainer = AAETrainer(config)
    vocab = trainer.get_vocabulary(train_data)
    model = AAE(vocab, config).to(args.device)

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

    # 打印GPU信息（如果使用CUDA）
    if args.device.startswith('cuda') and torch.cuda.is_available():
        gpu_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        print(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.1f} MB")

    print(f"Batch size: {config.n_batch}")
    print(f"Learning rate: {config.lr}")
    print(f"Pretrain epochs: {config.pretrain_epochs} (autoencoder only)")
    print(f"Train epochs: {config.train_epochs} (adversarial training)")
    print(f"Expected total steps: ~{total_steps} ({pretrain_steps} + {train_steps})")
    print("="*60 + "\n")

    if config.pretrain_epochs > 0:
        print("⚡ Phase 1: Pretraining autoencoder...")
        print(f"   This will train for {config.pretrain_epochs} epochs without discriminator")
        print(f"   Helps stabilize the model before adversarial training\n")

    try:
        model = trainer.fit(model, train_data, val_data=test_data)

        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, 'model_final.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"\n✅ Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    except RuntimeError as e:
        print(f"\n❌ Training failed with error: {e}")
        if "CUDA" in str(e) or "out of memory" in str(e):
            print("\n💡 解决建议:")
            print("1. 降低batch size: --n_batch 256 或 --n_batch 128")
            print("2. 清理GPU缓存: nvidia-smi 查看GPU使用情况")
            print("3. 尝试使用CPU: --device cpu")
            print(f"\n当前配置: batch_size={config.n_batch}, device={args.device}")
        raise

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel checkpoints saved in: {args.save_dir}")
    print(f"Log file: {config.log_file}")
    print("\nNext steps:")
    print("1. Generate samples: python scripts/sample.py aae \\")
    print(f"   --model_load {final_model_path} \\")
    print(f"   --config_load {config.config_save} \\")
    print(f"   --vocab_load {config.vocab_save} \\")
    print("   --n_samples 10000 --gen_save ./results/aae_generated_10k.csv")
    print("2. Evaluate metrics: python evaluate_baseline.py \\")
    print("   --input ./results/aae_generated_10k.csv \\")
    print("   --output ./results/aae_baseline_metrics.txt")


if __name__ == '__main__':
    main()
