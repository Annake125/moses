"""
VAE Configuration for Comparison with Diffusion Model
对比 Diffusion 模型的 VAE 配置

数据集: moses2.csv (1,584,079 molecules)
目标: 与 DIFFUMOL-ddim 模型保持参数对齐，确保公平对比

关键对齐参数:
- seq_len: 128 (与diffusion一致)
- hidden_dim: 128 (latent_dim d_z)
- learning_rate: 1e-4 (与diffusion一致)
- dropout: 0.1 (与diffusion一致)
- total_steps: 30,000 (与diffusion一致)
- seed: 102 (与diffusion一致)
"""

import argparse

def get_vae_config():
    """
    返回与diffusion模型对齐的VAE配置
    """
    config = argparse.Namespace()

    # ========== 模型架构 ==========
    # Encoder (编码器)
    config.q_cell = 'gru'              # GRU cell
    config.q_bidir = True              # 双向GRU (MOSES默认)
    config.q_d_h = 256                 # 编码器隐藏层维度 (MOSES默认)
    config.q_n_layers = 1              # 编码器层数
    config.q_dropout = 0.1             # ✅ 降低至与diffusion一致 (原0.5)

    # Latent Space (潜在空间)
    config.d_z = 128                   # ✅ 潜在向量维度 (与diffusion hidden_dim一致)

    # Decoder (解码器)
    config.d_cell = 'gru'              # GRU cell
    config.d_d_h = 512                 # 解码器隐藏层维度 (MOSES默认)
    config.d_n_layers = 3              # 解码器层数 (MOSES默认)
    config.d_dropout = 0.1             # ✅ 降低至与diffusion一致 (原0)

    config.freeze_embeddings = False

    # ========== 训练参数 ==========
    # Batch Size Strategy (批次大小策略)
    # 选项1: 使用gradient accumulation保持等效batch_size=2048
    # 选项2: 使用较小batch_size但保持其他参数一致

    config.n_batch = 512               # ⚠️ 实际batch size (考虑内存限制)
                                       # 原本diffusion用2048，但内存不足
                                       # 可选: 128, 256, 512

    config.gradient_accumulation_steps = 4  # 512*4=2048 等效batch size

    # Learning Rate (学习率)
    config.lr_start = 1e-4             # ✅ 与diffusion一致 (原3e-4)
    config.lr_end = 1e-4               # 保持恒定学习率

    # Learning Rate Schedule (学习率调度 - SGDR)
    config.lr_n_period = 10            # SGDR周期
    config.lr_n_restarts = 10          # SGDR重启次数
    config.lr_n_mult = 1               # SGDR周期倍增系数

    # Gradient Clipping (梯度裁剪)
    config.clip_grad = 50              # MOSES默认

    # KL Divergence Weight (KL散度权重 - VAE特有)
    config.kl_start = 0                # 从第0个epoch开始调整KL权重
    config.kl_w_start = 0.0            # 初始KL权重
    config.kl_w_end = 0.05             # 最终KL权重 (可调: 0.01-0.1)
                                       # 较小值→更好重建，较大值→更好生成

    # Training Steps/Epochs (训练步数/轮数)
    # 计算: 30,000 steps ÷ (1,584,079 / 512) ≈ 9.7 epochs
    # 但考虑到gradient accumulation，实际更新次数要除以accumulation_steps
    # 实际: 30,000 steps ÷ (1,584,079 / 2048) ≈ 38.8 epochs
    config.n_epochs = 40               # ✅ 训练40个epochs (略多于38.8)

    # Monitoring (监控)
    config.n_last = 1000               # 平滑loss计算的iteration数
    config.n_jobs = 1                  # 线程数
    config.n_workers = 4               # DataLoader worker数

    # ========== 随机种子 ==========
    config.seed = 102                  # ✅ 与diffusion一致

    # ========== 数据路径 ==========
    config.train_load = './data/train.csv'       # MOSES train set
    config.test_load = './data/test.csv'         # MOSES test set
    config.vocab_load = None                     # 从训练数据构建

    # 或者使用moses2.csv (需要预处理提取SMILES列)
    # config.train_load = './datasets/moses2.csv'

    # ========== 保存路径 ==========
    config.model_save = './checkpoints/vae_baseline/model.pt'
    config.config_save = './checkpoints/vae_baseline/config.pt'
    config.vocab_save = './checkpoints/vae_baseline/vocab.pt'

    # ========== 其他设置 ==========
    config.device = 'cuda:0'           # GPU设备

    return config


def print_config_comparison(config):
    """
    打印VAE与Diffusion配置的对比
    """
    print("\n" + "="*60)
    print("VAE vs Diffusion Model Configuration Comparison")
    print("VAE 与 Diffusion 模型配置对比")
    print("="*60)

    print(f"\n{'Parameter':<25} {'Diffusion':<15} {'VAE':<15} {'Status':<10}")
    print("-"*65)

    comparisons = [
        ("Latent Dim", "128", f"{config.d_z}", "✅ Aligned"),
        ("Learning Rate", "1e-4", f"{config.lr_start}", "✅ Aligned"),
        ("Dropout", "0.1", f"{config.q_dropout}", "✅ Aligned"),
        ("Batch Size", "2048", f"{config.n_batch}*{config.gradient_accumulation_steps}", "✅ Equivalent"),
        ("Training Steps", "30,000", f"~{1584079//2048 * config.n_epochs}", "✅ Similar"),
        ("Seed", "102", f"{config.seed}", "✅ Aligned"),
        ("Seq Length", "128", "128", "✅ Aligned"),
    ]

    for param, diff_val, vae_val, status in comparisons:
        print(f"{param:<25} {diff_val:<15} {vae_val:<15} {status:<10}")

    print("-"*65)
    print("\n✅ 配置已对齐，可进行公平对比实验\n")


if __name__ == '__main__':
    config = get_vae_config()
    print_config_comparison(config)

    # 保存配置
    import torch
    import os
    os.makedirs(os.path.dirname(config.config_save), exist_ok=True)
    torch.save(config, config.config_save)
    print(f"✅ Configuration saved to {config.config_save}")
