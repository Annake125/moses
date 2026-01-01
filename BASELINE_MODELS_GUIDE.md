# MOSES Baseline 模型完整指南

本文档提供VAE、CharRNN、AAE三个baseline模型的完整训练、采样和评估流程，用于与DIFFUMOL进行对比实验。

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [样本生成](#样本生成)
- [结果评估](#结果评估)
- [参数对比](#参数对比)
- [完整流程示例](#完整流程示例)

## 环境配置

```bash
# 确保已安装所需依赖
pip install torch pandas rdkit moses-chem tqdm
```

## 数据准备

确保moses2.csv数据集已准备好：

```bash
ls -lh ./data/moses2.csv
```

数据集应包含三个split：
- `train` - 训练集
- `test` - 测试集
- `test_scaffolds` - 测试scaffold集

## 模型训练

### 1. VAE (Variational Autoencoder)

```bash
nohup python train_vae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:1 \
    --seed 102 \
    --n_batch 256 > train_vae_baseline.log 2>&1 &
```

**关键参数：**
- Latent dimension: 128
- Learning rate: 1e-4
- Dropout: 0.1
- Batch size: 256 (可调整为128/512)

**输出：**
- 模型: `./checkpoints/vae_baseline/model_final.pt`
- 配置: `./checkpoints/vae_baseline/config.pt`
- 词汇表: `./checkpoints/vae_baseline/vocab.pt`

---

### 2. CharRNN (Character-level RNN)

```bash
nohup python train_charrnn_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --seed 102 \
    --n_batch 256 > train_charrnn_baseline.log 2>&1 &
```

**关键参数：**
- Hidden size: 768
- Num layers: 3
- Learning rate: 1e-3 (CharRNN需要较高学习率)
- Dropout: 0.1
- Batch size: 256

**输出：**
- 模型: `./checkpoints/charrnn_baseline/model_final.pt`
- 配置: `./checkpoints/charrnn_baseline/config.pt`
- 词汇表: `./checkpoints/charrnn_baseline/vocab.pt`

---

### 3. AAE (Adversarial Autoencoder)

```bash
nohup python train_aae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --seed 102 \
    --n_batch 512 \
    --lr 1e-4 > train_aae_baseline.log 2>&1 &
```

**关键参数：**
- Latent size: 128
- Encoder hidden: 256
- Decoder hidden: 512
- Learning rate: 1e-4
- Dropout: 0.1
- Batch size: 512 (可调整为256/1024)

**输出：**
- 模型: `./checkpoints/aae_baseline/model_final.pt`
- 配置: `./checkpoints/aae_baseline/config.pt`
- 词汇表: `./checkpoints/aae_baseline/vocab.pt`

---

## 样本生成

所有模型使用统一的采样脚本 `scripts/sample.py`：

### VAE 采样

```bash
python scripts/sample.py vae \
    --model_load ./checkpoints/vae_baseline/model_final.pt \
    --config_load ./checkpoints/vae_baseline/config.pt \
    --vocab_load ./checkpoints/vae_baseline/vocab.pt \
    --n_samples 10000 \
    --max_len 100 \
    --gen_save ./results/vae_generated_10k.csv \
    --device cuda:1
```

### CharRNN 采样

```bash
python scripts/sample.py char_rnn \
    --model_load ./checkpoints/charrnn_baseline/model_final.pt \
    --config_load ./checkpoints/charrnn_baseline/config.pt \
    --vocab_load ./checkpoints/charrnn_baseline/vocab.pt \
    --n_samples 10000 \
    --max_len 100 \
    --gen_save ./results/charrnn_generated_10k.csv \
    --device cuda:1
```

### AAE 采样

```bash
python scripts/sample.py aae \
    --model_load ./checkpoints/aae_baseline/model_final.pt \
    --config_load ./checkpoints/aae_baseline/config.pt \
    --vocab_load ./checkpoints/aae_baseline/vocab.pt \
    --n_samples 10000 \
    --max_len 100 \
    --gen_save ./results/aae_generated_10k.csv \
    --device cuda:1
```

**采样参数说明：**
- `--n_samples`: 生成的分子数量
- `--max_len`: 最大SMILES长度
- `--gen_save`: 输出CSV文件路径
- `--device`: 使用的设备 (cuda:0/cuda:1/cpu)

---

## 结果评估

所有模型使用统一的评估脚本 `evaluate_baseline.py`：

### VAE 评估

```bash
nohup python evaluate_baseline.py \
    --input ./results/vae_generated_10k.csv \
    --output ./results/vae_baseline_metrics.txt \
    --data_path ./data/moses2.csv \
    --device cpu \
    --n_jobs 8 > vae_eval.log 2>&1 &
```

### CharRNN 评估

```bash
nohup python evaluate_baseline.py \
    --input ./results/charrnn_generated_10k.csv \
    --output ./results/charrnn_baseline_metrics.txt \
    --data_path ./data/moses2.csv \
    --device cpu \
    --n_jobs 8 > charrnn_eval.log 2>&1 &
```

### AAE 评估

```bash
nohup python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt \
    --data_path ./data/moses2.csv \
    --device cpu \
    --n_jobs 8 > aae_eval.log 2>&1 &
```

**评估参数说明：**
- `--input`: 生成的分子CSV文件
- `--output`: 评估结果输出文件（.txt）
- `--data_path`: moses2.csv数据集路径
- `--model_name`: 模型名称（可选，自动检测）
- `--device`: 推荐使用`cpu`避免CUBLAS错误
- `--n_jobs`: 并行计算进程数

**输出文件：**
评估脚本会自动生成三个文件：
- `.txt` - 文本格式结果
- `.json` - JSON格式结果
- `.csv` - CSV格式结果（便于对比）

---

## 参数对比

### 与Diffusion对齐的参数

| 参数 | VAE | CharRNN | AAE | Diffusion |
|------|-----|---------|-----|-----------|
| **潜在维度** | 128 | - | 128 | 128 |
| **学习率** | 1e-4 | 1e-3* | 1e-4 | 1e-4 |
| **Dropout** | 0.1 | 0.1 | 0.1 | 0.1 |
| **Batch Size** | 256 | 256 | 512 | 2048 |
| **随机种子** | 102 | 102 | 102 | 102 |
| **目标训练步数** | ~30k | ~30k | ~30k | 30k |

*注：CharRNN使用1e-3学习率是因为RNN模型通常需要较高学习率

### 模型架构对比

| 模型 | 类型 | 核心机制 | 优势 | 劣势 |
|------|------|----------|------|------|
| **VAE** | 变分自编码器 | KL散度约束潜在空间 | 稳定训练，平滑潜在空间 | 生成质量可能受KL collapse影响 |
| **CharRNN** | 字符级RNN | 序列到序列生成 | 简单直接，训练快速 | 难以控制生成，容易产生无效分子 |
| **AAE** | 对抗自编码器 | 对抗训练约束潜在空间 | 潜在空间更灵活，生成质量好 | 训练不稳定，需要调节判别器 |

---

## 完整流程示例

### 并行训练所有模型

```bash
# 在不同GPU上并行训练三个模型
nohup python train_vae_baseline.py --use_moses2 --moses2_path ./data/moses2.csv --device cuda:0 --seed 102 > train_vae.log 2>&1 &
nohup python train_charrnn_baseline.py --use_moses2 --moses2_path ./data/moses2.csv --device cuda:1 --seed 102 > train_charrnn.log 2>&1 &
nohup python train_aae_baseline.py --use_moses2 --moses2_path ./data/moses2.csv --device cuda:0 --seed 102 > train_aae.log 2>&1 &

# 监控训练进度
tail -f train_vae.log
tail -f train_charrnn.log
tail -f train_aae.log
```

### 批量生成样本

```bash
# 创建批量生成脚本
cat > generate_all_samples.sh << 'EOF'
#!/bin/bash

# VAE
python scripts/sample.py vae \
    --model_load ./checkpoints/vae_baseline/model_final.pt \
    --config_load ./checkpoints/vae_baseline/config.pt \
    --vocab_load ./checkpoints/vae_baseline/vocab.pt \
    --n_samples 10000 --max_len 100 \
    --gen_save ./results/vae_generated_10k.csv \
    --device cuda:0

# CharRNN
python scripts/sample.py char_rnn \
    --model_load ./checkpoints/charrnn_baseline/model_final.pt \
    --config_load ./checkpoints/charrnn_baseline/config.pt \
    --vocab_load ./checkpoints/charrnn_baseline/vocab.pt \
    --n_samples 10000 --max_len 100 \
    --gen_save ./results/charrnn_generated_10k.csv \
    --device cuda:0

# AAE
python scripts/sample.py aae \
    --model_load ./checkpoints/aae_baseline/model_final.pt \
    --config_load ./checkpoints/aae_baseline/config.pt \
    --vocab_load ./checkpoints/aae_baseline/vocab.pt \
    --n_samples 10000 --max_len 100 \
    --gen_save ./results/aae_generated_10k.csv \
    --device cuda:0
EOF

chmod +x generate_all_samples.sh
./generate_all_samples.sh
```

### 批量评估

```bash
# 创建批量评估脚本
cat > evaluate_all_baselines.sh << 'EOF'
#!/bin/bash

# VAE
nohup python evaluate_baseline.py \
    --input ./results/vae_generated_10k.csv \
    --output ./results/vae_baseline_metrics.txt \
    > vae_eval.log 2>&1 &

# CharRNN
nohup python evaluate_baseline.py \
    --input ./results/charrnn_generated_10k.csv \
    --output ./results/charrnn_baseline_metrics.txt \
    > charrnn_eval.log 2>&1 &

# AAE
nohup python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt \
    > aae_eval.log 2>&1 &
EOF

chmod +x evaluate_all_baselines.sh
./evaluate_all_baselines.sh
```

### 对比结果

```bash
# 查看所有模型的评估结果
echo "=== VAE Results ==="
cat ./results/vae_baseline_metrics.txt

echo "=== CharRNN Results ==="
cat ./results/charrnn_baseline_metrics.txt

echo "=== AAE Results ==="
cat ./results/aae_baseline_metrics.txt

# 或者查看CSV格式便于对比
paste -d, \
    ./results/vae_baseline_metrics.csv \
    ./results/charrnn_baseline_metrics.csv \
    ./results/aae_baseline_metrics.csv
```

---

## 监控和调试

### 查看训练进度

```bash
# 实时查看训练日志
tail -f train_vae_baseline.log
tail -f train_charrnn_baseline.log
tail -f train_aae_baseline.log

# 查看GPU使用情况
nvidia-smi -l 1

# 查看训练日志文件
cat ./checkpoints/vae_baseline/log.txt
cat ./checkpoints/charrnn_baseline/log.txt
cat ./checkpoints/aae_baseline/log.txt
```

### 常见问题排查

**1. CUDA内存不足**
```bash
# 降低batch size
--n_batch 128  # 或更小
```

**2. CUBLAS错误（评估时）**
```bash
# 使用CPU进行评估
--device cpu
```

**3. 训练中断**
```bash
# 检查是否有checkpoint可以恢复
ls -lh ./checkpoints/*/model_*.pt
```

---

## 文件结构

```
moses/
├── train_vae_baseline.py          # VAE训练脚本
├── train_charrnn_baseline.py      # CharRNN训练脚本
├── train_aae_baseline.py          # AAE训练脚本
├── evaluate_baseline.py           # 通用评估脚本
├── BASELINE_MODELS_GUIDE.md       # 本文档
├── AAE_BASELINE_COMMANDS.md       # AAE详细命令
│
├── scripts/
│   └── sample.py                  # 通用采样脚本
│
├── data/
│   └── moses2.csv                 # 数据集
│
├── checkpoints/
│   ├── vae_baseline/
│   │   ├── model_final.pt
│   │   ├── config.pt
│   │   ├── vocab.pt
│   │   └── log.txt
│   ├── charrnn_baseline/
│   │   ├── model_final.pt
│   │   ├── config.pt
│   │   ├── vocab.pt
│   │   └── log.txt
│   └── aae_baseline/
│       ├── model_final.pt
│       ├── config.pt
│       ├── vocab.pt
│       └── log.txt
│
└── results/
    ├── vae_generated_10k.csv
    ├── vae_baseline_metrics.txt
    ├── vae_baseline_metrics.json
    ├── vae_baseline_metrics.csv
    ├── charrnn_generated_10k.csv
    ├── charrnn_baseline_metrics.txt
    ├── charrnn_baseline_metrics.json
    ├── charrnn_baseline_metrics.csv
    ├── aae_generated_10k.csv
    ├── aae_baseline_metrics.txt
    ├── aae_baseline_metrics.json
    └── aae_baseline_metrics.csv
```

---

## 评估指标说明

评估脚本会计算以下MOSES标准指标：

### 基础指标
- **Validity**: 生成的有效SMILES比例
- **Uniqueness**: 去重后的唯一分子比例
- **Novelty**: 不在训练集中的新分子比例

### 分布指标
- **FCD (Fréchet ChemNet Distance)**: 化学空间分布相似度
- **SNN (Scaffold Nearest Neighbor)**: Scaffold相似度
- **Frag**: Fragment相似度
- **Scaf**: Scaffold相似度

### 分子性质
- **LogP**: 分配系数
- **SA Score**: 合成可行性评分
- **QED**: 类药性评分
- **NP Score**: 天然产物相似度
- **Weight**: 分子量

---

## 下一步：与Diffusion对比

```bash
# 1. 确保Diffusion模型也生成了10k样本
# 2. 使用相同的评估脚本评估Diffusion结果
python evaluate_baseline.py \
    --input ./results/diffusion_generated_10k.csv \
    --output ./results/diffusion_metrics.txt

# 3. 对比所有模型的结果
python -c "
import pandas as pd

models = ['vae', 'charrnn', 'aae', 'diffusion']
dfs = []
for model in models:
    df = pd.read_csv(f'./results/{model}_baseline_metrics.csv')
    df.insert(0, 'Model', model.upper())
    dfs.append(df)

comparison = pd.concat(dfs, ignore_index=True)
print(comparison.to_string())
comparison.to_csv('./results/all_models_comparison.csv', index=False)
"
```

---

## 参考资料

- MOSES论文: https://arxiv.org/abs/1811.12823
- Moses库: https://github.com/molecularsets/moses
- RDKit文档: https://www.rdkit.org/docs/

---

最后更新: 2026-01-01
