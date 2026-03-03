"""
Section 4.4: Task별 적응적 정보 전이 메커니즘 분석

AITM(PITM) 모델의 Cross-Attention Weight를 분석하여
Target Task(T=22)가 Source Tasks(T=1, T=68, T=149)의 정보를
IPO별로 어떻게 적응적으로 참조하는지 실증 분석합니다.

분석 흐름:
    1. Seed별 AITM 체크포인트 생성 (python main.py benchmark --model aitm)
    2. Test 데이터에 대한 Attention Weight 추출
    3. 군집 분류 및 군집별 SHAP 분석 (seed별)
    4. 5개 seed 결과 평균/표준편차 집계
  5. 시각화 및 결과 저장

실행: python main.py attention
사전 조건: output/preprocess/preprocessed_data.csv 존재

출력:
  - output/figures/attention_weight_distribution.png
  - output/figures/attention_scatter_clusters.png
  - output/figures/attention_cluster_shap_comparison.png
  - output/attention_analysis_results.json
"""
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib
except ImportError:
    pass
import shap
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_RANDOM_STATE
from src.utils import set_seed
from src.models.aitm import AITMClassifier
from src.data_pipeline import IPODataPipeline


# ============================================================
# 설정
# ============================================================
CHECKPOINT_PATH = 'output/checkpoints/aitm_checkpoint.pt'
OUTPUT_DIR = 'output'
FIGURE_DIR = 'output/figures/attention'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAP_BACKGROUND_SAMPLES = 100
SHAP_EXPLAIN_SAMPLES = 100
N_ANALYSIS_SEEDS = 5


# ============================================================
# 1. 데이터 및 모델 로드
# ============================================================
def load_model_and_data():
    """
    저장된 AITM 체크포인트와 데이터를 로드하고,
    동일한 전처리를 적용하여 Train/Test 데이터를 준비합니다.

    Returns:
        model: 복원된 AITMClassifier
        X_train: 전처리된 Train 피처 (DataFrame)
        X_test: 전처리된 Test 피처 (DataFrame)
        y_test: Test 타겟 라벨 (Series)
        feature_names: 피처 이름 리스트
    """
    # 데이터 로드 및 전처리 (학습 시와 동일한 파이프라인)
    pipeline = IPODataPipeline()
    pipeline.load()
    shap_features = pipeline.load_shap_features()

    X_train_raw, X_test_raw, y_train_combined, y_test_combined = pipeline.get_train_test()
    X_train, X_test = pipeline.process(X_train_raw, X_test_raw, shap_features=shap_features)

    y_test = y_test_combined['Y']

    feature_names = X_test.columns.tolist()
    return X_train, X_test, y_test, feature_names


def build_seed_list(n_seeds=N_ANALYSIS_SEEDS, base_seed=MODEL_RANDOM_STATE):
    return [base_seed + i for i in range(n_seeds)]


def checkpoint_path_for_seed(seed):
    return os.path.join(OUTPUT_DIR, 'checkpoints', f'aitm_checkpoint_seed{seed}.pt')


def ensure_seed_checkpoints(seed_list):
    """train_evaluate.py를 seed별로 호출해 체크포인트 생성."""
    child_env = os.environ.copy()
    child_env.setdefault('MKL_THREADING_LAYER', 'GNU')
    child_env.setdefault('MKL_SERVICE_FORCE_INTEL', '1')

    for seed in seed_list:
        ckpt_path = checkpoint_path_for_seed(seed)
        cmd = [
            sys.executable,
            'main.py',
            'benchmark',
            '--model', 'aitm',
            '--seed', str(seed),
            '--ckpt-suffix', f'_seed{seed}',
        ]
        print(f"\n  [Seed {seed}] Training AITM via: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=child_env)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"체크포인트 생성 실패: {ckpt_path}")


def run_seeded_analysis(X_train, X_test, feature_names, seed_list):
    """Seed별 체크포인트를 로드해 Attention/SHAP 계산 후 집계."""
    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test

    source_names_ref = None
    attn_stack = []
    shap_stack = None
    shap_raw_by_seed = {}

    for seed in seed_list:
        ckpt_path = checkpoint_path_for_seed(seed)
        model, saved_feature_names = AITMClassifier.load_checkpoint(ckpt_path)

        if saved_feature_names != feature_names:
            raise ValueError(
                f"Seed {seed} 체크포인트 피처 불일치: "
                f"checkpoint={len(saved_feature_names)}, pipeline={len(feature_names)}"
            )

        attn_weights, source_names = extract_attention(model, X_test)
        if source_names_ref is None:
            source_names_ref = source_names
            shap_stack = {name: [] for name in source_names_ref}
        elif source_names != source_names_ref:
            raise ValueError(f"Seed {seed} source_names 불일치: {source_names} vs {source_names_ref}")

        attn_stack.append(attn_weights)
        cluster_labels_seed, _ = classify_clusters(attn_weights, source_names_ref)

        shap_raw_by_seed[str(seed)] = {}
        for name in source_names_ref:
            mask = cluster_labels_seed == name
            if mask.sum() < 5:
                print(f"  [Warning][Seed {seed}] {name} has only {mask.sum()} samples. SHAP may be unstable.")
            print(f"  [Seed {seed}] Computing SHAP for {name} (N={mask.sum()})...")

            if mask.sum() > 0:
                shap_vals = run_cluster_shap(model, X_train, X_test_arr[mask], feature_names)
            else:
                shap_vals = np.zeros(len(feature_names))

            shap_stack[name].append(shap_vals)
            shap_raw_by_seed[str(seed)][name] = shap_vals.tolist()

    attn_tensor = np.stack(attn_stack, axis=0)
    attn_mean = np.mean(attn_tensor, axis=0)
    attn_std = np.std(attn_tensor, axis=0)
    cluster_labels_mean, cluster_info_mean = classify_clusters(attn_mean, source_names_ref)

    shap_mean = {name: np.mean(np.vstack(vals), axis=0) for name, vals in shap_stack.items()}
    shap_std = {name: np.std(np.vstack(vals), axis=0) for name, vals in shap_stack.items()}

    return source_names_ref, attn_mean, attn_std, cluster_labels_mean, cluster_info_mean, shap_mean, shap_std, shap_raw_by_seed


# ============================================================
# 2. Attention Weight 추출
# ============================================================
def extract_attention(model, X_test):
    """
    Test 데이터에 대한 Target Task의 Attention Weight 추출.

    Returns:
        attn_weights: numpy array (N, n_sources)
        source_names: list of str (예: ['T=1', 'T=68', 'T=149'])
    """
    attn_weights, source_names = model.extract_attention_weights(X_test)

    print(f"\n  Attention Weight Statistics:")
    for i, name in enumerate(source_names):
        w = attn_weights[:, i]
        print(f"    {name}: mean={w.mean():.4f}, std={w.std():.4f}, "
              f"min={w.min():.4f}, max={w.max():.4f}")

    # Softmax 검증: 각 샘플의 weight 합 ≈ 1
    row_sums = attn_weights.sum(axis=1)
    print(f"    Weight sum check: mean={row_sums.mean():.6f}, "
          f"std={row_sums.std():.6f} (expected ~= 1.0)")

    return attn_weights, source_names


# ============================================================
# 3. Source Task 기반 분류
# ============================================================
def classify_clusters(attn_weights, source_names):
    """
    Argmax 기반 분류:
    각 IPO 샘플에서 가장 높은 Attention Weight를 받은 Source Task를 기준으로 분류.

    Args:
        attn_weights: (N, n_sources) attention weight matrix
        source_names: source task 이름 리스트

    Returns:
        cluster_labels: numpy array of source names (e.g., 'T=1', 'T=68', 'T=149')
        cluster_info: dict with cluster statistics
    """
    argmax_idx = np.argmax(attn_weights, axis=1)  # 각 샘플의 max weight source index
    cluster_labels = np.array([source_names[i] for i in argmax_idx])

    cluster_info = {}
    n_total = len(cluster_labels)
    for name in source_names:
        n_count = int((cluster_labels == name).sum())
        cluster_info[name] = {
            'description': f'{name} 최대 가중치',
            'count': n_count,
            'ratio': round(n_count / n_total * 100, 1) if n_total > 0 else 0.0,
            'mean_weights': {
                sn: float(attn_weights[cluster_labels == name, i].mean())
                for i, sn in enumerate(source_names)
            } if n_count > 0 else {},
        }

    return cluster_labels, cluster_info


# ============================================================
# 4. Source Task별 SHAP 분석
# ============================================================
def run_cluster_shap(model, X_background, X_cluster, feature_names):
    """
    특정 집단에 대한 SHAP 중요도 계산.

    Args:
        model: trained AITMClassifier
        X_background: background data (Train set)
        X_cluster: 군집 내 Test 데이터
        feature_names: 피처 이름 리스트

    Returns:
        importance: numpy array of mean absolute SHAP values
    """
    # Background 샘플링 (K-means)
    bg_data = X_background.values if hasattr(X_background, 'values') else X_background
    if len(bg_data) > SHAP_BACKGROUND_SAMPLES:
        background = shap.kmeans(bg_data, SHAP_BACKGROUND_SAMPLES)
    else:
        background = bg_data

    def predict_fn(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)

    # 군집 샘플링
    cluster_data = X_cluster.values if hasattr(X_cluster, 'values') else X_cluster
    sample_size = min(SHAP_EXPLAIN_SAMPLES, len(cluster_data))
    X_sample = cluster_data[:sample_size]

    shap_values = explainer.shap_values(X_sample, nsamples=100, silent=True)

    if isinstance(shap_values, list):
        vals = np.abs(shap_values[0]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)

    return vals


# ============================================================
# 5. 시각화
# ============================================================
def plot_attention_distribution(attn_weights, source_names, save_dir):
    """Source Task별 Attention Weight 분포 히스토그램"""
    fig, axes = plt.subplots(1, len(source_names), figsize=(5 * len(source_names), 5))
    if len(source_names) == 1:
        axes = [axes]

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    for i, (ax, name) in enumerate(zip(axes, source_names)):
        weights = attn_weights[:, i]
        ax.hist(weights, bins=25, alpha=0.8, color=colors[i % len(colors)],
                edgecolor='white', linewidth=0.5)
        ax.axvline(weights.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean={weights.mean():.3f}')
        ax.set_xlabel('Attention Weight', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Target Task(T=22)의 Source Task별 Attention Weight 분포',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'attention_weight_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_attention_scatter(attn_weights, source_names, cluster_labels, save_dir):
    """Attention Weight 2D Scatter Plot (첫 번째와 두 번째 파라미터 기준)"""
    if len(source_names) >= 2:
        plt.figure(figsize=(8, 8))
        
        x_idx, y_idx = 0, 1
        x_name, y_name = source_names[x_idx], source_names[y_idx]
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
        
        for i, name in enumerate(source_names):
            mask = cluster_labels == name
            plt.scatter(attn_weights[mask, x_idx], attn_weights[mask, y_idx],
                        c=colors[i % len(colors)], alpha=0.6, s=50, edgecolors='white', linewidth=0.5,
                        label=f'{name} 최대 가중치 (N={mask.sum()})')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Equal boundary')
        
        plt.xlabel(f'Attention Weight → {x_name}', fontsize=12)
        plt.ylabel(f'Attention Weight → {y_name}', fontsize=12)
        plt.title('IPO별 Attention Weight 분포 및 최대 가중치 Task', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'attention_scatter_clusters.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved: {save_path}")


def plot_cluster_shap_comparison(shap_results, feature_names, cluster_info, save_dir, top_k=10):
    """Source Task별 SHAP 중요도 비교 (side-by-side bar chart)"""
    n_sources = len(shap_results)
    fig, axes = plt.subplots(1, n_sources, figsize=(7 * n_sources, 8))
    if n_sources == 1:
        axes = [axes]
        
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    
    for i, (name, shap_vals) in enumerate(shap_results.items()):
        ax = axes[i]
        
        if np.sum(shap_vals) == 0:
            ax.set_title(f"{name}\n(No samples)", fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
            
        indices = np.argsort(shap_vals)[::-1][:top_k]
        top_features = [feature_names[j] for j in indices]
        top_vals = shap_vals[indices]

        y_pos = range(top_k)
        ax.barh(y_pos, top_vals[::-1], color=colors[i % len(colors)], alpha=0.8, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1], fontsize=10)
        ax.set_xlabel('Mean |SHAP value|', fontsize=11)

        info = cluster_info[name]
        ax.set_title(
            f"{info['description']}\n(N={info['count']}, {info['ratio']}%)",
            fontsize=12, fontweight='bold'
        )
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Source Task별 SHAP 기반 핵심 변수 중요도 비교',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'attention_cluster_shap_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {save_path}")


# ============================================================
# 6. 결과 출력 및 저장
# ============================================================
def print_comparison_table(shap_results, feature_names, cluster_info, top_k=10):
    """Source Task별 SHAP 중요도 비교표 콘솔 출력"""
    print("\n" + "=" * 100)
    print("  Attention-based Task Analysis: SHAP Feature Importance Comparison (5-Seed Mean)")
    print("=" * 100)

    for name in shap_results.keys():
        info = cluster_info[name]
        print(f"\n  {name}: {info['description']} (N={info['count']}, {info['ratio']}%)")
        if info['mean_weights']:
            weights_str = ", ".join([f"{k}={v:.4f}" for k, v in info['mean_weights'].items()])
            print(f"    Mean attention weights: {weights_str}")

    headers = [f"{'순위':<6}"]
    for name in shap_results.keys():
        headers.append(f"{name:<30}{'SHAP':<10}")
    print("\n" + "".join(headers))
    # print("-" * (6 + 40 * len(shap_results)))  # This handles the length nicely usually, but lets use static for simplicity
    print("-" * 100)

    # Get sorted indices for each source
    sorted_indices = {name: np.argsort(vals)[::-1][:top_k] for name, vals in shap_results.items()}

    for rank in range(top_k):
        row = [f"{rank+1:<6}"]
        for name, vals in shap_results.items():
            idx = sorted_indices[name]
            if rank < len(idx) and np.sum(vals) > 0:
                feat = feature_names[idx[rank]]
                val = vals[idx[rank]]
                row.append(f"{feat:<30}{val:<10.4f}")
            else:
                row.append(f"{'-':<30}{'-':<10}")
        print("".join(row))

    print("=" * 100)


def save_results(attn_weights, attn_std, source_names, cluster_labels, cluster_info,
                 shap_results_mean, shap_results_std, shap_raw_by_seed,
                 feature_names, seed_list):
    """분석 결과를 JSON으로 저장"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Source Task별 attention 통계
    attn_stats = {}
    for i, name in enumerate(source_names):
        w = attn_weights[:, i]
        w_std = attn_std[:, i]
        attn_stats[name] = {
            'mean': float(w.mean()),
            'std': float(w.std()),
            'mean_of_seed_std': float(w_std.mean()),
            'max_seed_std': float(w_std.max()),
            'min': float(w.min()),
            'max': float(w.max()),
        }

    # 군집별 Top-K 피처
    top_k = 10
    top_features = {}
    for name, vals in shap_results_mean.items():
        idx = np.argsort(vals)[::-1][:top_k]
        top_features[f"{name}_top_features"] = [
            {'rank': r + 1, 'feature': feature_names[i], 'shap_importance': float(vals[i])}
            for r, i in enumerate(idx) if np.sum(vals) > 0
        ]

    # 시각화용 Raw Data (Scatter plot용)
    visualization_data = {
        'scatter_data': {
            'x_axis': source_names[0] if len(source_names) > 0 else None,
            'y_axis': source_names[1] if len(source_names) > 1 else None,
            'points': [
                {
                    'sample_id': idx,
                    'cluster': str(cluster_labels[idx]),
                    'x': float(attn_weights[idx, 0]) if len(source_names) > 0 else 0.0,
                    'y': float(attn_weights[idx, 1]) if len(source_names) > 1 else 0.0,
                    'weights': {sn: float(attn_weights[idx, i]) for i, sn in enumerate(source_names)}
                }
                for idx in range(len(cluster_labels))
            ]
        }
    }

    result = {
        'seed_analysis': {
            'n_seeds': len(seed_list),
            'seed_list': seed_list,
            'base_seed': MODEL_RANDOM_STATE,
        },
        'attention_statistics': attn_stats,
        'cluster_info': {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in cluster_info.items()
        },
        'shap_mean_by_seed': {
            name: [float(v) for v in vals]
            for name, vals in shap_results_mean.items()
        },
        'shap_std_by_seed': {
            name: [float(v) for v in vals]
            for name, vals in shap_results_std.items()
        },
        'shap_raw_by_seed': shap_raw_by_seed,
        **top_features,
        'visualization_data': visualization_data,
        'total_samples': len(attn_weights),
    }

    json_path = os.path.join(OUTPUT_DIR, 'attention_analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  -> Saved results: {json_path}")


# ============================================================
# Main
# ============================================================
def main():
    set_seed(MODEL_RANDOM_STATE)

    print("=" * 60)
    print("  Section 4.4: Task별 적응적 정보 전이 메커니즘 분석")
    print("=" * 60)

    seed_list = build_seed_list()

    # 1. 데이터 로드
    print("\n[Step 1] Loading data...")
    X_train, X_test, y_test, feature_names = load_model_and_data()

    # 2. seed별 체크포인트 생성
    print(f"\n[Step 2] Training AITM checkpoints across {len(seed_list)} seeds...")
    ensure_seed_checkpoints(seed_list)

    # 3~4. seed별 분석 후 집계
    print("\n[Step 3-4] Running attention/SHAP analysis across seed checkpoints...")
    source_names, attn_weights, attn_std, cluster_labels, cluster_info, shap_results, shap_results_std, shap_raw_by_seed = run_seeded_analysis(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        seed_list=seed_list,
    )

    for key, info in cluster_info.items():
        print(f"  {key}: {info['description']} (N={info['count']}, {info['ratio']}%)")

    # 5. 시각화
    print("\n[Step 5] Generating visualizations...")
    os.makedirs(FIGURE_DIR, exist_ok=True)

    plot_attention_distribution(attn_weights, source_names, FIGURE_DIR)
    plot_attention_scatter(attn_weights, source_names, cluster_labels, FIGURE_DIR)
    plot_cluster_shap_comparison(shap_results, feature_names, cluster_info, FIGURE_DIR)

    # 6. 결과 출력 및 저장
    print_comparison_table(shap_results, feature_names, cluster_info)
    save_results(attn_weights, attn_std, source_names, cluster_labels, cluster_info,
                 shap_results, shap_results_std, shap_raw_by_seed,
                 feature_names, seed_list)

    print("\n[Done] Attention analysis complete.")


if __name__ == '__main__':
    main()
