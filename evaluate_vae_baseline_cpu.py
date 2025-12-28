"""
VAE Baseline评估脚本 - CPU版本，避免CUBLAS错误
直接从CSV读取数据进行评估
"""
import pandas as pd
import moses
from rdkit import Chem
import json
import os

print("="*60)
print("VAE Baseline Evaluation (CPU Mode - No CUBLAS errors)")
print("="*60)

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 1. 读取生成的分子
print("\n1. Loading generated molecules...")
df_gen = pd.read_csv('./results/vae_generated_10k.csv')
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
print("\n3. Loading test data from moses2.csv...")
df_data = pd.read_csv('data/moses2.csv')

# 提取test和train数据
test_smiles = df_data[df_data['SPLIT'] == 'test']['SMILES'].tolist()
train_smiles = df_data[df_data['SPLIT'] == 'train']['SMILES'].tolist()
test_scaffolds_smiles = df_data[df_data['SPLIT'] == 'test_scaffolds']['SMILES'].tolist()

print(f"   Test set: {len(test_smiles)} molecules")
print(f"   Train set: {len(train_smiles)} molecules")
print(f"   Test scaffolds: {len(test_scaffolds_smiles)} molecules")

# 4. 调用MOSES评估 - 使用CPU避免CUBLAS错误
print("\n4. Computing MOSES metrics (using CPU)...")
print("   (This may take several minutes on CPU...)")

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
        device='cpu',  # ✅ 使用CPU避免CUBLAS错误
        n_jobs=8
    )

    # 5. 显示结果
    print("\n" + "="*60)
    print("VAE Baseline Evaluation Results")
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
        'model': 'VAE_Baseline',
        'validity': validity,
        **metrics
    }

    # 保存为JSON
    with open('results/vae_baseline_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 保存为TXT
    with open('results/vae_baseline_metrics.txt', 'w') as f:
        f.write("VAE Baseline Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"{'Metric':<30s} {'Value':<15s}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Validity':<30s} {validity:<15.4f}\n")
        for key in sorted(metrics.keys()):
            f.write(f"{key:<30s} {metrics[key]:<15.4f}\n")
        f.write("="*60 + "\n")

    # 保存为CSV
    df_results = pd.DataFrame([results])
    df_results.to_csv('results/vae_baseline_metrics.csv', index=False)

    print("\n✅ Results saved to:")
    print("   - results/vae_baseline_metrics.json")
    print("   - results/vae_baseline_metrics.txt")
    print("   - results/vae_baseline_metrics.csv")

except Exception as e:
    print(f"\n❌ Full evaluation failed: {e}")
    print("\nTrying simplified evaluation (without scaffolds)...")

    try:
        # 备用方案：不使用test_scaffolds
        metrics = moses.get_all_metrics(
            gen=valid_smiles,
            test=test_smiles,
            train=train_smiles,
            device='cpu',  # ✅ 使用CPU
            n_jobs=8
        )

        print("\n" + "="*60)
        print("VAE Baseline Results (Simplified - No Scaffolds)")
        print("="*60)
        print(f"{'Metric':<30s} {'Value':<15s}")
        print("-"*60)
        print(f"{'Validity':<30s} {validity:<15.4f}")

        for key in sorted(metrics.keys()):
            print(f"{key:<30s} {metrics[key]:<15.4f}")

        print("="*60)

        # 保存简化结果
        results = {
            'model': 'VAE_Baseline',
            'validity': validity,
            'note': 'Simplified evaluation without scaffold metrics',
            **metrics
        }

        with open('results/vae_baseline_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        with open('results/vae_baseline_metrics.txt', 'w') as f:
            f.write("VAE Baseline Results (Simplified)\n")
            f.write("="*60 + "\n")
            f.write(f"{'Metric':<30s} {'Value':<15s}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Validity':<30s} {validity:<15.4f}\n")
            for key in sorted(metrics.keys()):
                f.write(f"{key:<30s} {metrics[key]:<15.4f}\n")
            f.write("="*60 + "\n")

        df_results = pd.DataFrame([results])
        df_results.to_csv('results/vae_baseline_metrics.csv', index=False)

        print("\n✅ Simplified results saved to:")
        print("   - results/vae_baseline_metrics.json")
        print("   - results/vae_baseline_metrics.txt")
        print("   - results/vae_baseline_metrics.csv")

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
            'model': 'VAE_Baseline',
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'note': 'Minimal evaluation - only basic metrics computed'
        }

        print("\n" + "="*60)
        print("VAE Baseline Results (Minimal)")
        print("="*60)
        print(f"{'Validity':<30s} {validity:<15.4f}")
        print(f"{'Uniqueness':<30s} {uniqueness:<15.4f}")
        print(f"{'Novelty':<30s} {novelty:<15.4f}")
        print("="*60)

        with open('results/vae_baseline_metrics_minimal.json', 'w') as f:
            json.dump(minimal_results, f, indent=2)

        print("\n✅ Minimal results saved to:")
        print("   - results/vae_baseline_metrics_minimal.json")

print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60)
