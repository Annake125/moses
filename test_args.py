"""
快速测试train_vae_baseline.py的参数解析
不需要安装依赖
"""

import argparse
import sys

def test_argparse():
    """测试参数解析"""
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
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=102,
                        help='Random seed')

    # 模型保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints/vae_baseline',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='./checkpoints/vae_baseline/log.txt',
                        help='Log file path')

    # 测试命令行
    test_args = [
        '--use_moses2',
        '--moses2_path', './data/moses2.csv',
        '--device', 'cuda:0',
        '--n_batch', '512',
        '--seed', '102'
    ]

    try:
        args = parser.parse_args(test_args)
        print("✅ 参数解析成功!")
        print("\n解析结果:")
        print(f"  use_moses2: {args.use_moses2}")
        print(f"  moses2_path: {args.moses2_path}")
        print(f"  device: {args.device}")
        print(f"  n_batch: {args.n_batch}")
        print(f"  seed: {args.seed}")
        print(f"  save_dir: {args.save_dir}")
        print(f"  log_file: {args.log_file}")
        return True
    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        return False

if __name__ == '__main__':
    success = test_argparse()
    sys.exit(0 if success else 1)
