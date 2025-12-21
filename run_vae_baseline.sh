#!/bin/bash

# VAE Baseline 训练脚本 - 一键启动
# 与Diffusion模型参数对齐

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "  VAE Baseline Training for Comparison with Diffusion Model"
echo "  VAE Baseline训练 - 与Diffusion模型对比"
echo "======================================================================"
echo ""

# ==================== 配置参数 ====================

# GPU设备
DEVICE="cuda:0"

# Batch size (根据内存调整: 128/256/512/1024)
BATCH_SIZE=512

# 随机种子 (与diffusion对齐)
SEED=102

# 数据集选择
USE_MOSES2=true  # true=使用moses2.csv, false=使用MOSES官方数据集
MOSES2_PATH="./datasets/moses2.csv"

# 保存路径
SAVE_DIR="./checkpoints/vae_baseline"
LOG_FILE="$SAVE_DIR/training.log"

# ==================== 检查环境 ====================

echo "🔍 Checking environment..."

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.7+"
    exit 1
fi

echo "✅ Python version: $(python --version)"

# 检查GPU
if [ "$DEVICE" != "cpu" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  nvidia-smi not found, falling back to CPU"
        DEVICE="cpu"
    else
        echo "✅ GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
        echo "   Available memory:"
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | xargs -I {} echo "   {}MiB free"
    fi
fi

# 检查数据集
if [ "$USE_MOSES2" = true ]; then
    if [ ! -f "$MOSES2_PATH" ]; then
        echo "❌ moses2.csv not found at: $MOSES2_PATH"
        echo "   Please check the path or set USE_MOSES2=false to use MOSES official dataset"
        exit 1
    fi
    echo "✅ Dataset: moses2.csv found at $MOSES2_PATH"
    # 检查文件大小
    FILE_SIZE=$(du -h "$MOSES2_PATH" | cut -f1)
    echo "   File size: $FILE_SIZE"
else
    echo "✅ Dataset: Will use MOSES official dataset (auto-download if needed)"
fi

# 创建保存目录
mkdir -p "$SAVE_DIR"
echo "✅ Save directory: $SAVE_DIR"

echo ""

# ==================== 显示配置 ====================

echo "======================================================================"
echo "  Training Configuration"
echo "  训练配置"
echo "======================================================================"
echo ""
printf "%-25s %s\n" "Device:" "$DEVICE"
printf "%-25s %s\n" "Batch Size:" "$BATCH_SIZE"
printf "%-25s %s\n" "Random Seed:" "$SEED"
printf "%-25s %s\n" "Dataset:" "$([ "$USE_MOSES2" = true ] && echo "moses2.csv" || echo "MOSES official")"
printf "%-25s %s\n" "Save Directory:" "$SAVE_DIR"
printf "%-25s %s\n" "Log File:" "$LOG_FILE"
echo ""

echo "======================================================================"
echo "  Parameters Aligned with Diffusion Model"
echo "  与Diffusion模型对齐的参数"
echo "======================================================================"
echo ""
printf "%-25s %-15s %-15s %s\n" "Parameter" "Diffusion" "VAE" "Status"
echo "----------------------------------------------------------------------"
printf "%-25s %-15s %-15s %s\n" "Latent Dim" "128" "128" "✅ Aligned"
printf "%-25s %-15s %-15s %s\n" "Learning Rate" "1e-4" "1e-4" "✅ Aligned"
printf "%-25s %-15s %-15s %s\n" "Dropout" "0.1" "0.1" "✅ Aligned"
printf "%-25s %-15s %-15s %s\n" "Batch Size" "2048" "$BATCH_SIZE" "⚠️  Memory limit"
printf "%-25s %-15s %-15s %s\n" "Training Steps" "30,000" "~30,000" "✅ Aligned"
printf "%-25s %-15s %-15s %s\n" "Seed" "102" "$SEED" "✅ Aligned"
echo "----------------------------------------------------------------------"
echo ""

# ==================== 确认启动 ====================

read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "======================================================================"
echo "  Starting Training..."
echo "  开始训练..."
echo "======================================================================"
echo ""

# ==================== 构建命令 ====================

CMD="python train_vae_baseline.py \
    --device $DEVICE \
    --n_batch $BATCH_SIZE \
    --seed $SEED \
    --save_dir $SAVE_DIR \
    --log_file $LOG_FILE"

if [ "$USE_MOSES2" = true ]; then
    CMD="$CMD --use_moses2 --moses2_path $MOSES2_PATH"
fi

# 显示完整命令
echo "Running command:"
echo "$CMD"
echo ""

# ==================== 执行训练 ====================

# 保存输出到日志文件同时显示在终端
$CMD 2>&1 | tee "${SAVE_DIR}/console_output.log"

# ==================== 训练完成 ====================

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "  ✅ Training Complete!"
    echo "  ✅ 训练完成!"
    echo "======================================================================"
    echo ""
    echo "Model saved to: $SAVE_DIR"
    echo "Logs saved to: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Generate samples:"
    echo "   python scripts/sample.py vae \\"
    echo "       --model_load $SAVE_DIR/model_final.pt \\"
    echo "       --vocab_load $SAVE_DIR/vocab.pt \\"
    echo "       --config_load $SAVE_DIR/config.pt \\"
    echo "       --n_samples 30000 \\"
    echo "       --gen_save ./generated/vae_samples.csv"
    echo ""
    echo "2. Evaluate metrics:"
    echo "   python scripts/eval.py \\"
    echo "       --ref_path ./data/test.csv \\"
    echo "       --gen_path ./generated/vae_samples.csv"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "  ❌ Training Failed"
    echo "  ❌ 训练失败"
    echo "======================================================================"
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "- Out of memory: Try reducing BATCH_SIZE (e.g., 256 or 128)"
    echo "- Missing dependencies: pip install torch pandas tqdm rdkit"
    echo "- Data not found: Check MOSES2_PATH or set USE_MOSES2=false"
    echo ""
    exit 1
fi
