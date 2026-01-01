# AAE Baseline 训练和评估命令

本文档提供AAE (Adversarial Autoencoder) baseline模型的完整训练、采样和评估命令。

## 环境要求

```bash
# 确保已安装所需依赖
pip install torch pandas rdkit moses-chem tqdm
```

## 1. 训练AAE模型

### 使用moses2数据集训练

```bash
# 使用GPU训练（推荐）
nohup python train_aae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --seed 102 \
    --n_batch 512 \
    --lr 1e-4 > train_aae_baseline.log 2>&1 &

# 如果GPU内存不足，可以降低batch size
nohup python train_aae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --seed 102 \
    --n_batch 256 \
    --lr 1e-4 > train_aae_baseline.log 2>&1 &

# 使用CPU训练（较慢）
nohup python train_aae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cpu \
    --seed 102 \
    --n_batch 256 \
    --lr 1e-4 > train_aae_baseline.log 2>&1 &
```

### 参数说明

- `--use_moses2`: 使用moses2.csv数据集
- `--moses2_path`: moses2.csv文件路径
- `--device`: 训练设备 (cuda:0, cuda:1, 或 cpu)
- `--seed`: 随机种子 (102与diffusion对齐)
- `--n_batch`: 批次大小 (256/512/1024，根据GPU内存调整)
- `--lr`: 学习率 (1e-4与diffusion对齐)
- `--save_dir`: 模型保存目录 (默认: ./checkpoints/aae_baseline)

### 监控训练进度

```bash
# 查看训练日志
tail -f train_aae_baseline.log

# 查看GPU使用情况
nvidia-smi -l 1
```

## 2. 生成分子样本

训练完成后，使用以下命令生成10,000个分子样本：

```bash
python scripts/sample.py aae \
    --model_load ./checkpoints/aae_baseline/model_final.pt \
    --config_load ./checkpoints/aae_baseline/config.pt \
    --vocab_load ./checkpoints/aae_baseline/vocab.pt \
    --n_samples 10000 \
    --max_len 100 \
    --gen_save ./results/aae_generated_10k.csv \
    --device cuda:0
```

### 参数说明

- `aae`: 模型类型（AAE）
- `--model_load`: 训练好的模型权重路径
- `--config_load`: 模型配置文件路径
- `--vocab_load`: 词汇表文件路径
- `--n_samples`: 生成样本数量
- `--max_len`: 最大SMILES长度
- `--gen_save`: 输出CSV文件路径
- `--device`: 使用的设备

## 3. 评估生成质量

使用通用评估脚本评估生成的分子：

```bash
# 后台运行评估（推荐）
nohup python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt \
    --data_path ./data/moses2.csv \
    --model_name AAE \
    --device cpu \
    --n_jobs 8 > aae_eval.log 2>&1 &

# 前台运行评估
python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt \
    --data_path ./data/moses2.csv \
    --model_name AAE \
    --device cpu \
    --n_jobs 8
```

### 参数说明

- `--input`: 生成的分子CSV文件路径
- `--output`: 评估结果输出文件路径（.txt）
- `--data_path`: moses2.csv数据集路径
- `--model_name`: 模型名称（用于报告，可选，会自动检测）
- `--device`: 计算设备（推荐使用cpu避免CUBLAS错误）
- `--n_jobs`: 并行计算的进程数

### 输出文件

评估脚本会生成三个文件：
- `./results/aae_baseline_metrics.txt` - 文本格式结果
- `./results/aae_baseline_metrics.json` - JSON格式结果
- `./results/aae_baseline_metrics.csv` - CSV格式结果

## 4. 完整流程示例

```bash
# Step 1: 训练AAE模型
nohup python train_aae_baseline.py \
    --use_moses2 \
    --moses2_path ./data/moses2.csv \
    --device cuda:0 \
    --seed 102 > train_aae_baseline.log 2>&1 &

# 等待训练完成，可以使用 tail -f train_aae_baseline.log 监控

# Step 2: 生成10k个分子样本
python scripts/sample.py aae \
    --model_load ./checkpoints/aae_baseline/model_final.pt \
    --config_load ./checkpoints/aae_baseline/config.pt \
    --vocab_load ./checkpoints/aae_baseline/vocab.pt \
    --n_samples 10000 \
    --max_len 100 \
    --gen_save ./results/aae_generated_10k.csv \
    --device cuda:0

# Step 3: 评估生成质量
nohup python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt > aae_eval.log 2>&1 &
```

## 5. 与其他Baseline模型对比

```bash
# VAE
python evaluate_baseline.py \
    --input ./results/vae_generated_10k.csv \
    --output ./results/vae_baseline_metrics.txt

# CharRNN
python evaluate_baseline.py \
    --input ./results/charrnn_generated_10k.csv \
    --output ./results/charrnn_baseline_metrics.txt

# AAE
python evaluate_baseline.py \
    --input ./results/aae_generated_10k.csv \
    --output ./results/aae_baseline_metrics.txt
```

## 6. 关键配置参数对比

| 参数 | AAE | VAE | CharRNN | Diffusion |
|------|-----|-----|---------|-----------|
| Latent Size | 128 | 128 | - | 128 |
| Learning Rate | 1e-4 | 1e-4 | 1e-3 | 1e-4 |
| Dropout | 0.1 | 0.1 | 0.1 | 0.1 |
| Batch Size | 512 | 256 | 256 | 2048 |
| Seed | 102 | 102 | 102 | 102 |
| Target Steps | ~30k | ~30k | ~30k | 30k |

## 7. 常见问题

### GPU内存不足

```bash
# 降低batch size
python train_aae_baseline.py --n_batch 256  # 或 128
```

### CUBLAS错误

```bash
# 评估时使用CPU
python evaluate_baseline.py --device cpu
```

### 查看训练进度

```bash
tail -f train_aae_baseline.log
tail -f ./checkpoints/aae_baseline/log.txt
```

## 8. 文件结构

```
moses/
├── train_aae_baseline.py         # AAE训练脚本
├── train_vae_baseline.py         # VAE训练脚本
├── train_charrnn_baseline.py     # CharRNN训练脚本
├── evaluate_baseline.py          # 通用评估脚本
├── scripts/
│   └── sample.py                 # 通用采样脚本
├── checkpoints/
│   └── aae_baseline/
│       ├── model_final.pt        # 最终模型权重
│       ├── config.pt             # 模型配置
│       ├── vocab.pt              # 词汇表
│       └── log.txt               # 训练日志
└── results/
    ├── aae_generated_10k.csv     # 生成的分子
    ├── aae_baseline_metrics.txt  # 评估结果
    ├── aae_baseline_metrics.json # JSON格式结果
    └── aae_baseline_metrics.csv  # CSV格式结果
```
