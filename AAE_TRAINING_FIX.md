# AAE训练问题诊断和修复

## 问题症状

AAE生成的分子质量极差：
- **Validity**: 0.3865 (38.65%) - 应该 >90%
- **QED**: 0.0589 - 极低
- 大量无效SMILES

## 根本原因分析

### 错误的配置（之前）

为了与diffusion模型对齐，我过度降低了AAE的训练参数：

| 参数 | 错误配置 | 应该配置 | 影响 |
|------|---------|---------|------|
| `lr` | 1e-4 | 1e-3 | ❌ 学习率太低，AAE无法收敛 |
| `train_epochs` | 10 | 50-120 | ❌ 严重训练不足 (只有8%) |
| `encoder_hidden_size` | 256 | 512 | ❌ 模型容量不足 |
| `pretrain_epochs` | 0 | 10+ | ❌ 没有预训练，对抗训练不稳定 |
| `encoder_dropout` | 0.1 | 0 | ❌ AAE对dropout敏感 |
| `decoder_dropout` | 0.1 | 0 | ❌ AAE对dropout敏感 |

### 为什么AAE需要特殊处理？

**AAE (Adversarial Autoencoder) 与 VAE 的关键区别：**

1. **对抗训练更难收敛**
   - VAE使用KL散度约束（稳定）
   - AAE使用判别器对抗训练（不稳定）
   - 需要更高学习率来平衡generator和discriminator

2. **需要预训练**
   - 先训练autoencoder部分（无判别器）
   - 再引入判别器进行对抗训练
   - 防止训练初期崩溃

3. **对dropout敏感**
   - 对抗训练中dropout会增加不稳定性
   - MOSES官方AAE不使用dropout

4. **需要更多训练轮数**
   - VAE: 10 epochs 可能足够
   - CharRNN: 10 epochs 足够
   - AAE: 至少50-120 epochs

## 修复方案

### 新配置（已修复）

```python
# AAE推荐配置
config.encoder_hidden_size = 512       # ✅ MOSES标准
config.encoder_dropout = 0             # ✅ 移除dropout
config.decoder_dropout = 0             # ✅ 移除dropout
config.latent_size = 128               # ✅ 与diffusion对齐
config.lr = 1e-3                       # ✅ AAE标准学习率
config.pretrain_epochs = 10            # ✅ 预训练稳定模型
config.train_epochs = 50               # ✅ 足够的训练（折中方案）
config.step_size = 20                  # ✅ 学习率衰减周期
```

### 训练阶段

**Phase 1: 预训练 (10 epochs)**
- 只训练autoencoder (encoder + decoder)
- 不使用discriminator
- 目的：让重构损失先收敛

**Phase 2: 对抗训练 (50 epochs)**
- 同时训练autoencoder和discriminator
- 判别器强制潜在编码符合先验分布
- 学习率会在第20、40 epochs衰减

## 重新训练命令

### 删除旧模型

```bash
rm -rf ./checkpoints/aae_baseline/*
```

### 重新训练（新配置）

```bash
nohup python train_aae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --seed 102 \
    --n_batch 512 \
    --lr 1e-3 \
    --pretrain_epochs 10 \
    --train_epochs 50 > train_aae_baseline.log 2>&1 &
```

**预期训练时间：**
- 数据量: ~1.58M分子
- Steps per epoch: 1.58M / 512 ≈ 3086
- 总训练步数: 3086 × (10 + 50) = 185,160 steps
- 预计时长: 3-5小时 (取决于GPU)

### 监控训练

```bash
# 实时查看日志
tail -f train_aae_baseline.log

# 查看训练损失
grep "loss" train_aae_baseline.log | tail -20

# 查看GPU使用
nvidia-smi -l 1
```

### 重新生成样本

训练完成后：

```bash
# 删除旧样本
rm -f ./results/aae_generated_10k.csv

# 重新生成
python scripts/sample.py aae \
    --model_load ./checkpoints/aae_baseline/model_final.pt \
    --config_load ./checkpoints/aae_baseline/config.pt \
    --vocab_load ./checkpoints/aae_baseline/vocab.pt \
    --n_samples 10000 \
    --max_len 100 \
    --gen_save ./results/aae_generated_10k.csv \
    --device cuda:0

# 重新评估
nohup python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt > aae_eval.log 2>&1 &
```

## 预期结果

使用新配置，AAE应该达到：

| 指标 | 之前 | 预期 | 参考（MOSES论文） |
|------|------|------|------------------|
| **Validity** | 0.3865 | **>0.90** | 0.936 |
| **Uniqueness** | 1.0000 | ~0.998 | 0.998 |
| **Novelty** | 0.9972 | ~0.90 | 0.895 |
| **FCD/Test** | 4.3322 | **<3.0** | 2.576 |
| **QED** | 0.0589 | **>0.90** | 0.934 |

## 关键教训

### ❌ 不要过度追求参数对齐

虽然我们想要公平对比baseline和diffusion，但：
- 不同模型有不同的最优配置
- AAE需要遵循其自身的训练规律
- **强行对齐会导致训练失败**

### ✅ 正确的对比方法

**可以对齐的参数：**
- `latent_size`: 128 (模型容量)
- `seed`: 102 (可重现性)
- 数据集: moses2.csv (公平对比)

**不应该对齐的参数：**
- `lr`: 不同模型需要不同学习率
- `dropout`: 不同架构对dropout的需求不同
- `train_epochs`: 应该训练到收敛，而不是固定步数

### ✅ 验证训练质量

训练后检查：

1. **查看训练损失曲线**
   ```bash
   grep "loss" train_aae_baseline.log
   ```
   - 损失应该持续下降
   - 预训练阶段reconstruction loss应该收敛

2. **检查生成样本质量**
   ```bash
   head -20 ./results/aae_generated_10k.csv
   ```
   - 应该是有效的SMILES
   - 不应该有大量padding或无效字符

3. **对比MOSES benchmark**
   - 参考：https://github.com/molecularsets/moses
   - AAE validity应该 >93%

## 如果还是效果不好

### 选项1: 更多训练

```bash
--train_epochs 100  # 接近MOSES标准的120
```

### 选项2: 增加预训练

```bash
--pretrain_epochs 20  # 更充分的预训练
```

### 选项3: 调整判别器训练频率

修改 `train_aae_baseline.py`:
```python
config.discriminator_steps = 2  # 每步训练discriminator 2次
```

### 选项4: 使用更大的模型

```bash
--n_batch 256  # 降低batch size
# 手动修改config增加模型容量
```

## 总结

**问题**: AAE validity只有38%，远低于预期的90%+

**原因**: 训练严重不足（epochs太少、lr太低、没有预训练）

**解决**: 使用AAE标准配置（lr=1e-3, epochs=50, pretrain=10）

**下一步**: 重新训练AAE模型，预期validity >90%

---

更新时间: 2026-01-05
