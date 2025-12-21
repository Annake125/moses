# VAE Baseline训练指南 - 与Diffusion模型对比

## 📋 目标

训练MOSES VAE baseline模型，与"Hitting stride by degrees Fine grained molecular generation via diffusion"论文的DIFFUMOL-ddim模型进行公平对比实验。

## 🎯 推荐的Baseline模型组合

建议选择以下2-3个模型进行对比：

1. **VAE** (本指南) - 优先推荐
   - 基于连续潜在空间，与diffusion模型可比性强
   - 训练稳定，结果可靠
   - FCD: 0.099, Valid: 97.67%

2. **CharRNN** - 经典序列模型baseline
   - 最简单直接的生成模型
   - FCD: 0.0732 (MOSES中最好)
   - 训练快速，适合快速对比

3. **AAE** - 可选
   - 结合VAE和GAN
   - 展示不同潜在空间约束方法的差异

## 📊 数据集说明

### moses2.csv vs MOSES官方数据集

你的`moses2.csv`分析：
- ✅ **1,584,079行** = MOSES官方train set大小
- ✅ 包含train/test/test_scaffolds标准划分
- ✅ 额外列（qed, logp等）可用于后续分析
- ✅ **结论**: moses2.csv = MOSES + 额外属性，可直接使用

**推荐**: 使用moses2.csv，与你的diffusion模型保持一致

## ⚙️ 参数对齐策略

### 核心对齐参数

| 参数 | Diffusion | VAE (本配置) | 状态 | 说明 |
|------|-----------|-------------|------|------|
| **潜在空间维度** | hidden_dim=128 | d_z=128 | ✅ 完全对齐 | 核心可比参数 |
| **学习率** | 1e-4 | 1e-4 | ✅ 完全对齐 | 优化器设置 |
| **Dropout** | 0.1 | 0.1 | ✅ 完全对齐 | 正则化强度 |
| **序列长度** | 128 | 128 | ✅ 完全对齐 | SMILES最大长度 |
| **训练步数** | 30,000 | ~30,000 | ✅ 接近 | 总优化步数 |
| **随机种子** | 102 | 102 | ✅ 完全对齐 | 可复现性 |
| **Batch Size** | 2048 | **512** | ⚠️ 不同 | **内存限制** |

### Batch Size说明

**问题**: 你的内存只有10GB，batch_size=2048会OOM

**解决方案**:

1. **方案A: 使用batch_size=512** (推荐，简单)
   - 实际batch: 512
   - 在论文中说明：由于内存限制使用batch_size=512
   - 其他参数完全对齐，仍是有效对比

2. **方案B: Gradient Accumulation** (等效batch=2048，但需修改代码)
   - 实际batch: 256
   - Accumulation steps: 8
   - 等效batch: 256 × 8 = 2048
   - 需要修改trainer代码

**建议**: 先用方案A (batch=512)，简单且有效

### VAE特有参数

这些参数是VAE模型特有的，不影响与diffusion的可比性：

- **KL权重**: `kl_w_end=0.05`
  - 控制潜在空间正则化强度
  - 可调范围: 0.01-0.1
  - 较小值→更好重建，较大值→更好生成

- **编码器/解码器架构**:
  - 编码器: 双向GRU, hidden=256
  - 解码器: 3层GRU, hidden=512
  - 这是MOSES标准配置

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装MOSES
pip install torch pandas tqdm rdkit

# 或使用conda
conda install -c rdkit rdkit
pip install molsets
```

### 2. 准备数据

#### 选项A: 使用moses2.csv (推荐，与你的diffusion一致)

```bash
# 确保moses2.csv在正确位置
ls ./datasets/moses2.csv

# 训练
python train_vae_baseline.py \
    --use_moses2 \
    --moses2_path ./datasets/moses2.csv \
    --device cuda:0 \
    --n_batch 512 \
    --seed 102
```

#### 选项B: 使用MOSES官方数据集

```bash
# MOSES会自动下载数据集
python train_vae_baseline.py \
    --device cuda:0 \
    --n_batch 512 \
    --seed 102
```

### 3. 训练参数说明

```bash
python train_vae_baseline.py --help
```

主要参数：
- `--device`: GPU设备 (cuda:0/cuda:1/cpu)
- `--n_batch`: batch size (推荐512，内存充足可用1024)
- `--seed`: 随机种子 (使用102与diffusion对齐)
- `--save_dir`: 模型保存目录
- `--use_moses2`: 使用moses2.csv而非官方数据集

### 4. 训练过程监控

训练会显示：
```
Training (epoch #0): loss=4.12345 (kl=0.02345 recon=4.10000) klw=0.00000 lr=0.0001
```

- `loss`: 总损失 = kl_weight × kl + recon
- `kl`: KL散度损失 (潜在空间正则化)
- `recon`: 重建损失 (SMILES重建质量)
- `klw`: 当前KL权重 (逐渐从0增加到0.05)
- `lr`: 学习率

### 5. 生成分子样本

```bash
# 使用MOSES自带脚本
python scripts/sample.py vae \
    --model_load ./checkpoints/vae_baseline/model_final.pt \
    --vocab_load ./checkpoints/vae_baseline/vocab.pt \
    --config_load ./checkpoints/vae_baseline/config.pt \
    --n_samples 30000 \
    --gen_save ./generated/vae_samples.csv
```

### 6. 评估指标

```bash
# 评估生成分子质量
python scripts/eval.py \
    --ref_path ./data/test.csv \
    --gen_path ./generated/vae_samples.csv
```

会输出MOSES标准指标：
- **Valid**: 有效SMILES比例
- **Unique**: 唯一分子比例
- **FCD**: Fréchet ChemNet Distance (越低越好)
- **SNN**: 与测试集的相似度
- **Novelty**: 新颖分子比例
- 等等...

## 📈 预期训练时间

基于MOSES官方数据 (1.6M分子):

| Batch Size | GPU | 每epoch时间 | 总时间 (40 epochs) |
|-----------|-----|------------|-------------------|
| 512 | V100 | ~15分钟 | ~10小时 |
| 512 | RTX 3090 | ~20分钟 | ~13小时 |
| 256 | GTX 1080Ti | ~20分钟 | ~13小时 |

## 🔧 故障排查

### 内存不足 (OOM)

```bash
# 降低batch size
python train_vae_baseline.py --n_batch 256  # 或128
```

### 训练不稳定

- 检查KL权重: 如果loss震荡，降低`kl_w_end`
- 检查学习率: 如果loss不下降，可能需要调整lr

### 生成质量差

- 增加训练epochs
- 调整KL权重 (0.01-0.1范围)
- 检查数据质量

## 📝 训练检查清单

训练前确认：
- [ ] 数据集路径正确 (moses2.csv或MOSES官方)
- [ ] GPU内存充足 (推荐≥8GB for batch=512)
- [ ] 保存目录已创建
- [ ] 随机种子设置为102 (与diffusion对齐)

训练中监控：
- [ ] Loss正常下降
- [ ] KL loss逐渐增加 (随KL weight增加)
- [ ] Recon loss持续下降
- [ ] 定期保存checkpoint

训练后验证：
- [ ] 生成30k样本
- [ ] 计算MOSES metrics
- [ ] 与diffusion模型对比
- [ ] 检查valid/unique/FCD等指标

## 🎓 对比实验建议

### 公平对比的要点

1. **相同数据集**: 都用moses2.csv
2. **相同评估指标**: 都用MOSES标准metrics
3. **相同随机种子**: seed=102
4. **对齐关键参数**: latent_dim, lr, dropout等
5. **说明差异**: 在论文中说明batch_size不同的原因

### 实验报告建议

对比表格示例：

| Model | Valid↑ | Unique@10k↑ | FCD↓ | SNN↑ | Novelty↑ |
|-------|--------|-------------|------|------|----------|
| VAE | XX.XX% | X.XXXX | X.XXX | 0.XXX | 0.XXX |
| CharRNN | XX.XX% | X.XXXX | X.XXX | 0.XXX | 0.XXX |
| DIFFUMOL | XX.XX% | X.XXXX | X.XXX | 0.XXX | 0.XXX |

参数设置表：

| Parameter | VAE | CharRNN | DIFFUMOL | Note |
|-----------|-----|---------|----------|------|
| Latent Dim | 128 | - | 128 | ✅ Aligned |
| Learning Rate | 1e-4 | 1e-3 | 1e-4 | ✅ Aligned (VAE) |
| Batch Size | 512 | 64 | 2048 | ⚠️ Memory limit |
| Training Steps | ~30k | ~30k | 30k | ✅ Aligned |

## 📚 下一步

完成VAE训练后：

1. **训练CharRNN baseline**:
   ```bash
   python scripts/train.py char_rnn --train_load ./data/train.csv
   ```

2. **可选: 训练AAE baseline**:
   ```bash
   python scripts/train.py aae --train_load ./data/train.csv
   ```

3. **对比分析**:
   - 收集所有模型的metrics
   - 分析各模型的优劣
   - 绘制分布对比图 (logP, MW, QED等)

4. **论文撰写**:
   - 描述实验设置
   - 展示对比结果
   - 分析diffusion模型的优势

## ❓ 常见问题

**Q: 为什么不用batch_size=2048?**
A: 你的GPU内存只有10GB，batch=2048会OOM。batch=512已足够进行有效对比。

**Q: KL权重应该设多少?**
A: 默认0.05是MOSES标准值。如果生成质量差，可尝试0.01-0.1范围。

**Q: 训练多少epochs合适?**
A: 默认配置训练40 epochs ≈ 30k steps，与diffusion对齐。

**Q: 必须用seed=102吗?**
A: 不必须，但使用相同seed可以减少随机性影响，使对比更公平。

**Q: moses2.csv和官方MOSES数据集有区别吗?**
A: moses2.csv = MOSES + 额外化学性质列，分子数据一致，可以互换使用。

## 📞 需要帮助？

如遇问题，检查：
1. MOSES文档: https://github.com/molecularsets/moses
2. MOSES论文: https://arxiv.org/abs/1811.12823
3. 本仓库Issues

祝实验顺利！🎉
