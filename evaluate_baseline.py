"""
Baseline模型评估脚本 - 通用版本，支持命令行参数
直接从CSV读取生成的分子数据进行评估，避免CUBLAS错误

使用说明:
    python evaluate_baseline.py --input ./results/vae_generated_10k.csv --output ./results/vae_baseline_metrics.txt
    python evaluate_baseline.py --input ./results/charrnn_generated_10k.csv --output ./results/charrnn_baseline_metrics.txt
    python evaluate_baseline.py --input ./results/aae_generated_10k.csv --output ./results/aae_baseline_metrics.txt
"""
import pandas as pd
import moses
from rdkit import Chem
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baseline Models')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to generated molecules CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output metrics file (txt)')
    parser.add_argument('--data_path', type=str, default='./data/moses2.csv',
                        help='Path to moses2.csv dataset')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (e.g., VAE, CharRNN, AAE). Auto-detected if not specified.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda:0). Default: cpu to avoid CUBLAS errors')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Number of parallel jobs for metric computation')

    args = parser.parse_args()

    # 自动检测模型名称
    if args.model_name is None:
        if 'vae' in args.input.lower():
            args.model_name = 'VAE'
        elif 'charrnn' in args.input.lower() or 'char_rnn' in args.input.lower():
            args.model_name = 'CharRNN'
        elif 'aae' in args.input.lower():
            args.model_name = 'AAE'
        else:
            args.model_name = 'Unknown'

    print("="*60)
    print(f"{args.model_name} Baseline Evaluation (Device: {args.device})")
    print("="*60)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. 读取生成的分子
    print("\n1. Loading generated molecules...")
    print(f"   Input file: {args.input}")
    df_gen = pd.read_csv(args.input)
    gen_smiles = df_gen['SMILES'].tolist() if 'SMILES' in df_gen.columns else df_gen['smiles'].tolist()
    print(f"   Total generated: {len(gen_smiles)}")

    # 2. 计算Validity
    print("\n2. Calculating validity...")
    valid_smiles = []
    for smi in gen_smiles:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_smiles.append(str(smi))
        except:
            continue

    validity = len(valid_smiles) / len(gen_smiles) if gen_smiles else 0
    print(f"   Valid: {len(valid_smiles)}/{len(gen_smiles)} ({validity:.4f})")

    # 3. 读取测试数据（从moses2.csv）
    print(f"\n3. Loading test data from {args.data_path}...")
    df_data = pd.read_csv(args.data_path)

    # 提取test和train数据
    test_smiles = df_data[df_data['SPLIT'] == 'test']['SMILES'].tolist()
    train_smiles = df_data[df_data['SPLIT'] == 'train']['SMILES'].tolist()
    test_scaffolds_smiles = df_data[df_data['SPLIT'] == 'test_scaffolds']['SMILES'].tolist()

    print(f"   Test set: {len(test_smiles)} molecules")
    print(f"   Train set: {len(train_smiles)} molecules")
    print(f"   Test scaffolds: {len(test_scaffolds_smiles)} molecules")

    # 4. 调用MOSES评估
    print(f"\n4. Computing MOSES metrics (using {args.device})...")
    print("   (This may take several minutes...)")

    try:
        # 完整评估（包括scaffolds）
        print("\n   Trying full evaluation with scaffolds...")
        metrics = moses.get_all_metrics(
            gen=valid_smiles,
            test=test_smiles,
            train=train_smiles,
            test_scaffolds=test_scaffolds_smiles,
            ptest=None,  # 不使用预计算统计
            ptest_scaffolds=None,  # 不使用预计算统计
            device=args.device,
            n_jobs=args.n_jobs
        )

        # 5. 显示结果
        print("\n" + "="*60)
        print(f"{args.model_name} Baseline Evaluation Results")
        print("="*60)
        print(f"{'Metric':<30s} {'Value':<15s}")
        print("-"*60)
        print(f"{'Validity':<30s} {validity:<15.4f}")

        for key in sorted(metrics.keys()):
            value = metrics[key]
            print(f"{key:<30s} {value:<15.4f}")

        print("="*60)

        # 6. 保存结果
        results = {
            'model': f'{args.model_name}_Baseline',
            'validity': validity,
            **metrics
        }

        # 生成输出文件路径
        output_base = args.output.rsplit('.', 1)[0]  # 移除扩展名
        json_path = output_base + '.json'
        csv_path = output_base + '.csv'

        # 保存为JSON
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # 保存为TXT
        with open(args.output, 'w') as f:
            f.write(f"{args.model_name} Baseline Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"{'Metric':<30s} {'Value':<15s}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Validity':<30s} {validity:<15.4f}\n")
            for key in sorted(metrics.keys()):
                f.write(f"{key:<30s} {metrics[key]:<15.4f}\n")
            f.write("="*60 + "\n")

        # 保存为CSV
        df_results = pd.DataFrame([results])
        df_results.to_csv(csv_path, index=False)

        print("\n✅ Results saved to:")
        print(f"   - {json_path}")
        print(f"   - {args.output}")
        print(f"   - {csv_path}")

    except Exception as e:
        print(f"\n❌ Full evaluation failed: {e}")
        print("\nTrying simplified evaluation (without scaffolds)...")

        try:
            # 备用方案：不使用test_scaffolds
            metrics = moses.get_all_metrics(
                gen=valid_smiles,
                test=test_smiles,
                train=train_smiles,
                device=args.device,
                n_jobs=args.n_jobs
            )

            print("\n" + "="*60)
            print(f"{args.model_name} Baseline Results (Simplified - No Scaffolds)")
            print("="*60)
            print(f"{'Metric':<30s} {'Value':<15s}")
            print("-"*60)
            print(f"{'Validity':<30s} {validity:<15.4f}")

            for key in sorted(metrics.keys()):
                print(f"{key:<30s} {metrics[key]:<15.4f}")

            print("="*60)

            # 保存简化结果
            results = {
                'model': f'{args.model_name}_Baseline',
                'validity': validity,
                'note': 'Simplified evaluation without scaffold metrics',
                **metrics
            }

            output_base = args.output.rsplit('.', 1)[0]
            json_path = output_base + '.json'
            csv_path = output_base + '.csv'

            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)

            with open(args.output, 'w') as f:
                f.write(f"{args.model_name} Baseline Results (Simplified)\n")
                f.write("="*60 + "\n")
                f.write(f"{'Metric':<30s} {'Value':<15s}\n")
                f.write("-"*60 + "\n")
                f.write(f"{'Validity':<30s} {validity:<15.4f}\n")
                for key in sorted(metrics.keys()):
                    f.write(f"{key:<30s} {metrics[key]:<15.4f}\n")
                f.write("="*60 + "\n")

            df_results = pd.DataFrame([results])
            df_results.to_csv(csv_path, index=False)

            print("\n✅ Simplified results saved to:")
            print(f"   - {json_path}")
            print(f"   - {args.output}")
            print(f"   - {csv_path}")

        except Exception as e2:
            print(f"\n❌ Simplified evaluation also failed: {e2}")
            print("\nTrying minimal evaluation (basic metrics only)...")

            # 最小评估：只计算不依赖复杂GPU计算的指标
            from collections import Counter

            # Uniqueness
            unique_smiles = list(set(valid_smiles))
            uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

            # Novelty (不在训练集中)
            train_set = set(train_smiles)
            novel_smiles = [s for s in unique_smiles if s not in train_set]
            novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

            minimal_results = {
                'model': f'{args.model_name}_Baseline',
                'validity': validity,
                'uniqueness': uniqueness,
                'novelty': novelty,
                'note': 'Minimal evaluation - only basic metrics computed'
            }

            print("\n" + "="*60)
            print(f"{args.model_name} Baseline Results (Minimal)")
            print("="*60)
            print(f"{'Validity':<30s} {validity:<15.4f}")
            print(f"{'Uniqueness':<30s} {uniqueness:<15.4f}")
            print(f"{'Novelty':<30s} {novelty:<15.4f}")
            print("="*60)

            output_base = args.output.rsplit('.', 1)[0]
            minimal_json = output_base + '_minimal.json'

            with open(minimal_json, 'w') as f:
                json.dump(minimal_results, f, indent=2)

            print("\n✅ Minimal results saved to:")
            print(f"   - {minimal_json}")

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
