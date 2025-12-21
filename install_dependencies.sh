#!/bin/bash

# 快速安装MOSES及其依赖
# Quick installation script for MOSES and dependencies

set -e

echo "======================================================================"
echo "  MOSES Installation Script"
echo "  MOSES 依赖安装脚本"
echo "======================================================================"
echo ""

# 检查Python版本
echo "🔍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# 安装MOSES及其依赖
echo ""
echo "📦 Installing MOSES and dependencies..."
echo ""

# 选项1: 使用pip安装（推荐）
echo "Option 1: Installing from current directory (recommended)"
pip install -e .

# 或者选项2: 仅安装基础依赖
# echo "Option 2: Installing basic dependencies only"
# pip install torch pandas tqdm numpy scipy matplotlib seaborn

echo ""
echo "======================================================================"
echo "  ✅ Installation Complete!"
echo "======================================================================"
echo ""

# 验证安装
echo "🔍 Verifying installation..."
python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'  Pandas version: {pandas.__version__}')"
python -c "import moses; print(f'  MOSES installed successfully')" 2>/dev/null || echo "  ⚠️  MOSES module not found (this is OK if using scripts directly)"

echo ""
echo "You can now run the VAE training script:"
echo "  python train_vae_baseline.py --help"
echo ""
