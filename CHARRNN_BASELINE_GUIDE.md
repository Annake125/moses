# CharRNN Baseline训练指南 - 与Diffusion模型对比

## 📋 CharRNN简介

**CharRNN (Character-level Recurrent Neural Network)** 是最经典的序列生成模型，用于分子生成的baseline。

### 特点：
- ✅ **简单直接** - 最基础的序列生成方法
- ✅ **训练快速** - 比VAE/GAN更快收敛
- ✅ **性能优秀** - MOSES benchmark中FCD最低(0.0732)
- ✅ **适合对比** - 展示扩散模型相对经典方法的优势

### 模型架构：
```
输入SMILES → Embedding → 3层LSTM(hidden=768) → Softmax → 输出字符
```

## 🎯 与Diffusion模型的对比

### 参数对齐策略

| 参数 | Diffusion | CharRNN | 对齐状态 | 说明 |
|------|-----------|---------|---------|------|
| **模型类型** | Transformer | LSTM | ❌ 不同 | 架构本质不同 |
| **学习率** | 1e-4 | 1e-3 | ⚠️ 不同 | CharRNN需要较高LR |
| **Dropout** | 0.1 | 0.1 | ✅ 对齐 | 正则化强度 |
| **Batch Size** | 2048 | 256 | ⚠️ 不同 | 折中选择 |
| **训练步数** | 30,000 | ~30,000 | ✅ 对齐 | 总优化步数 |
| **随机种子** | 102 | 102 | ✅ 对齐 | 可复现性 |
| **数据集** | moses2.csv | moses2.csv | ✅ 对齐 | 相同数据 |

### 为什么学习率不同？

**CharRNN使用1e-3而非1e-4的原因**：
1. LSTM对学习率更敏感，需要较高值快速收敛
2. Transformer有残差连接和layer norm，可以用更小学习率
3. 这是各模型的**标准配置**，保持能展示各自最佳性能

**对比实验影响**：
- ✅ 仍然公平：每个模型用最适合自己的学习率
- ✅ 展示优势：如果diffusion用相同LR性能更好，说明模型优越性
- ✅ 实用价值：展示实际应用中各模型的表现

## 🚀 快速开始

### **方法1：一键启动（推荐）**

```bash
./run_charrnn_baseline.sh
```

### **方法2：手动运行**

```bash
# GPU训练
python train_charrnn_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --n_batch 256 \
    --lr 0.001 \
    --seed 102

# CPU训练（如果GPU有问题）
python train_charrnn_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cpu \
    --n_batch 512 \
    --lr 0.001 \
    --seed 102
```

## ⚙️ 配置参数说明

### **核心参数**

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `--device` | cuda:0 | 设备选择 | cuda:0/cpu |
| `--n_batch` | 256 | Batch大小 | 128-512 |
| `--lr` | 0.001 | 学习率 | 1e-4 ~ 1e-3 |
| `--seed` | 102 | 随机种子 | 102 (对齐) |

### **模型参数**（在代码中设置）

```python
# 模型架构
num_layers = 3        # LSTM层数
hidden = 768          # 隐藏层维度
dropout = 0.1         # Dropout率（与diffusion对齐）

# 训练设置
train_epochs = 10     # 训练轮数（约30k steps）
step_size = 10        # 学习率衰减周期
gamma = 0.5           # 学习率衰减因子
```

## 📊 预期训练时间

基于moses2.csv数据集 (1.58M molecules):

| 设备 | Batch Size | 每Epoch时间 | 总时间 (10 epochs) |
|------|-----------|------------|-------------------|
| V100 | 256 | ~8分钟 | ~1.3小时 |
| A40 | 256 | ~6分钟 | ~1小时 |
| RTX 3090 | 256 | ~10分钟 | ~1.7小时 |
| CPU | 512 | ~45分钟 | ~7.5小时 |

**CharRNN比VAE快约2-3倍！**

## 📈 训练过程监控

训练会显示：
```
Training (epoch 0): loss=2.45 lr=0.00100
```

- `loss`: 交叉熵损失（越低越好，通常收敛到<1.0）
- `lr`: 当前学习率（每10个epoch衰减50%）

**正常训练**：
- Epoch 0-5: loss快速下降 (2.5 → 1.5)
- Epoch 5-10: loss缓慢下降 (1.5 → 0.8)
- 最终loss: 0.6-1.0

## 🎯 生成样本

训练完成后生成分子：

```bash
# 使用MOSES官方脚本
python scripts/sample.py char_rnn \
    --model_load ./checkpoints/charrnn_baseline/model_final.pt \
    --vocab_load ./checkpoints/charrnn_baseline/vocab.pt \
    --config_load ./checkpoints/charrnn_baseline/config.pt \
    --n_samples 30000 \
    --gen_save ./generated/charrnn_samples.csv
```

## 📊 评估指标

```bash
# 使用CPU评估（避免CUBLAS错误）
python evaluate_vae_baseline_cpu.py  # 修改为读取charrnn_samples.csv

# 或使用MOSES官方评估
python scripts/eval.py \
    --ref_path ./data/test.csv \
    --gen_path ./generated/charrnn_samples.csv
```

### **MOSES官方Benchmark结果**

| 指标 | CharRNN | VAE | Diffusion (你的) |
|------|---------|-----|-----------------|
| Valid | 97.48% | 97.67% | ? |
| Unique@10k | 99.94% | 99.84% | ? |
| FCD | **0.0732** | 0.099 | ? |
| SNN | 0.6015 | 0.6257 | ? |
| Novelty | 84.19% | 69.49% | ? |

CharRNN的**FCD最低**（越低越好），说明生成分布与真实分布最接近！

## 🔧 故障排查

### **GPU CUBLAS错误**

如果遇到和VAE一样的CUBLAS错误：

```bash
# 使用CPU训练
python train_charrnn_baseline.py --device cpu --n_batch 512

# CharRNN在CPU上也很快！（比VAE快很多）
```

### **内存不足**

```bash
# 降低batch size
python train_charrnn_baseline.py --n_batch 128
```

### **训练不收敛**

```bash
# 尝试调整学习率
python train_charrnn_baseline.py --lr 0.0005  # 降低学习率
```

## 💡 优化建议

### **提高性能**

1. **增加epochs** - CharRNN可以训练更多轮（15-20 epochs）
2. **调整学习率** - 尝试1e-4到2e-3范围
3. **增大batch size** - 如果内存允许，用512或1024

### **加速训练**

1. **使用GPU** - 比CPU快5-10倍
2. **增大batch** - batch=512比256快约1.5倍
3. **减少验证** - 每5个epoch验证一次而非每次

## 📝 对比实验建议

### **完整对比表**

| 模型 | 架构 | Latent | LR | Batch | Steps | Valid | FCD | Novelty |
|------|------|--------|-----|-------|-------|-------|-----|---------|
| CharRNN | LSTM | - | 1e-3 | 256 | 30k | ? | ? | ? |
| VAE | GRU | 128 | 1e-4 | 256 | 30k | ? | ? | ? |
| DIFFUMOL | Transformer | 128 | 1e-4 | 2048 | 30k | ? | ? | ? |

### **实验报告模板**

```markdown
### Baseline Comparison

We compare DIFFUMOL with two classical baselines:

1. **CharRNN**: Character-level LSTM, the simplest baseline
   - Fast training (~1 hour)
   - Strong performance (FCD: 0.073 in MOSES benchmark)
   - Different learning rate (1e-3) suitable for LSTM

2. **VAE**: Variational autoencoder with continuous latent space
   - Latent dimension aligned with DIFFUMOL (128)
   - Same learning rate (1e-4) and dropout (0.1)
   - Comparable training time (~10 hours)

All models trained on the same dataset (MOSES, 1.58M molecules)
with aligned training steps (~30k) and random seed (102).
```

## 🎓 CharRNN vs VAE vs Diffusion

### **优劣对比**

| 特点 | CharRNN | VAE | Diffusion |
|------|---------|-----|-----------|
| **训练速度** | ⭐⭐⭐ 最快 | ⭐⭐ 中等 | ⭐ 较慢 |
| **内存占用** | ⭐⭐⭐ 最小 | ⭐⭐ 中等 | ⭐ 较大 |
| **生成质量** | ⭐⭐ 好 | ⭐⭐ 好 | ⭐⭐⭐ 最好？ |
| **可控性** | ⭐ 差 | ⭐⭐ 中等 | ⭐⭐⭐ 好 |
| **多样性** | ⭐⭐⭐ 高 | ⭐⭐ 中等 | ⭐⭐ 中等 |

### **适用场景**

- **CharRNN**: 快速baseline，概念验证
- **VAE**: 需要连续表示，插值生成
- **Diffusion**: 最高质量，可控生成

## ✅ 检查清单

训练前：
- [ ] 数据集路径正确 (moses2.csv)
- [ ] GPU可用（或准备用CPU）
- [ ] 保存目录已创建
- [ ] 随机种子设为102

训练中：
- [ ] Loss正常下降
- [ ] 每个epoch约6-10分钟（GPU）
- [ ] 定期保存checkpoint

训练后：
- [ ] 生成30k样本
- [ ] 计算MOSES metrics
- [ ] 与VAE和Diffusion对比

## 🎉 总结

CharRNN是**最简单、最快、最容易训练**的baseline：

✅ **优点**：
- 训练快（1小时 vs VAE的10小时）
- CPU友好（不受CUBLAS错误影响）
- 性能优秀（MOSES benchmark中表现最好）

⚠️ **注意**：
- 学习率与diffusion不同（但合理）
- 缺少潜在空间（无法做插值）
- 可控性较差

**建议训练顺序**：
1. **CharRNN** - 最快，先跑出结果
2. **VAE** - 中等复杂度，提供对比
3. **Diffusion** - 你的主模型

祝训练顺利！🚀
