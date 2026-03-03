"""
SHAP-based Recursive Feature Elimination (RFE) & Analysis

AITM 모델을 사용하여 반복적 피처 제거(RFE)를 수행합니다.
각 단계에서 피처를 제거하며 Validation AUC/F1 변화를 추적하여 최적의 피처 개수를 결정합니다.
(단순 Top-K 방식보다 논리적이고 과학적인 접근)

실행: python 7_shap_analysis.py --step 1
출력: 
  - output/shap_selected_features.json (최적 피처 목록)
  - output/figures/shap_rfe_performance.png (피처 개수 vs 성능 그래프)
  - output/figures/shap_summary_optimal.png (최적 모델의 SHAP 요약)
"""
import argparse
import json
import os
import sys
import copy
import time
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
from collections import Counter
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)



# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_RANDOM_STATE, MTL_MANUAL_SOURCE_TASKS, SCALER
)
from src.utils import set_seed, scale_features, evaluate_with_threshold
from src.models.aitm import AITMClassifier
from src.data_pipeline import IPODataPipeline


RFE_VAL_RATIO = 0.2  # Train 80% 내부에서 Val 비율
SHAP_BACKGROUND_SAMPLES = 100  # KernelExplainer 배경 샘플 수 (K-means)
SHAP_EXPLAIN_SAMPLES = 100     # SHAP 값 계산에 사용할 검증 샘플 수


def load_data():
    """
    전처리된 데이터 로드 (IPODataPipeline 사용).
    벤치마크와 동일한 80/20 시계열 분할 후 Train 80%만 사용합니다.
    Test 20%는 RFE에서 사용하지 않아 벤치마크 결과의 신뢰성을 보장합니다.

    Train 80% 내부에서 다시 시계열 80/20 분할하여 RFE Train/Val을 구성합니다.
    """
    pipeline = IPODataPipeline()
    pipeline.load()

    # 벤치마크와 동일한 80/20 시계열 분할
    X_train_all, _X_test, y_train_all_combined, _y_test = pipeline.get_train_test()

    # Train 80% 내부에서 시계열 RFE Train/Val 분할 (단일 분할: RFE 특성상 CV 미적용)
    X_train, X_val, y_train_combined, y_val_combined = pipeline.split_train_internal(
        X_train_all, y_train_all_combined, val_ratio=RFE_VAL_RATIO
    )

    y_train = y_train_combined['Y'].values
    y_val = y_val_combined['Y'].values

    # Source Tasks
    source_cols = [c for c in y_train_combined.columns if c.startswith('Y_T')]
    y_source_train = {col: y_train_combined[col].values for col in source_cols}

    return X_train, X_val, y_train, y_val, y_source_train


def train_aitm_subset(X_train_sub, y_train, y_source_train, X_val_sub, y_val, verbose=False):
    """특정 피처 서브셋으로 AITM 학습 및 평가"""
    # Scaling (이미 인코딩된 상태라고 가정)
    X_train_s, X_val_s = scale_features(X_train_sub, X_val_sub, scaler_type=SCALER)

    # y_dict 구성
    y_train_dict = {'target': y_train}
    for i, (col, val) in enumerate(y_source_train.items()):
        y_train_dict[f'source_{i}'] = val



    model = AITMClassifier(
        dropout=0.2, learning_rate=1e-3, batch_size=128, epochs=50,
        source_days=list(MTL_MANUAL_SOURCE_TASKS), verbose=verbose
    )

    model.fit(X_train_s, y_train_dict,
              X_valid=X_val_s, y_valid_dict={'target': y_val})

    # 성능 평가 (공통 함수 활용)
    y_pred_proba = model.predict_proba(X_val_s)[:, 1]
    metrics = evaluate_with_threshold(y_val, y_pred_proba)
    auc = metrics['AUC']
    f1 = metrics['F1']

    return model, X_train_s, X_val_s, auc, f1


def calculate_shap_importance(model, X_train, X_val):
    """SHAP 중요도 계산 (Global Mean Absolute SHAP)"""
    # Background (K-means for speed in loop)
    if len(X_train) > SHAP_BACKGROUND_SAMPLES:
        background = shap.kmeans(X_train, SHAP_BACKGROUND_SAMPLES)
    else:
        background = X_train

    def predict_fn(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)

    # Val set 샘플링
    sample_size = min(SHAP_EXPLAIN_SAMPLES, len(X_val))
    X_val_sample = X_val.iloc[:sample_size] if hasattr(X_val, 'iloc') else X_val[:sample_size]

    shap_values = explainer.shap_values(X_val_sample, nsamples=100, silent=True)

    # 중요도: 절대값 평균
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[0]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)

    return vals


def run_rfe(X_train, y_train, y_source_train, X_val, y_val, step=1, min_features=5, seed=42):
    """Recursive Feature Elimination Loop"""
    
    # 시드 설정 (매 반복마다 다른 시드 적용)
    set_seed(seed)

    # 1. 인코딩 먼저 수행 (고정된 피처 풀 생성, 스케일링은 아래에서 별도 수행)
    pipeline = IPODataPipeline()
    X_train_enc, X_val_enc = pipeline.process(X_train, X_val, scale=False)
    current_features = X_train_enc.columns.tolist()

    # 피처별 누적 중요도 (Global Importance Sum)
    history = []  # {'n_features': int, 'auc': float, 'f1': float, 'features': list}
    feature_importance_history = []
    
    print(f"\nStarting RFE (Total Features: {len(current_features)}) with Seed {seed}")
    
    # 초기 전체 피처에 대한 중요도 계산을 위해 미리 한 번 학습
    # (루프 진입 전 초기 상태 기록용)
    initial_model, X_tr_s, X_vl_s, initial_auc, initial_f1 = train_aitm_subset(
        X_train_enc, y_train, y_source_train, X_val_enc, y_val, verbose=False
    )
    
    X_tr_df = pd.DataFrame(X_tr_s, columns=current_features)
    X_vl_df = pd.DataFrame(X_vl_s, columns=current_features)
    initial_imp = calculate_shap_importance(initial_model, X_tr_df, X_vl_df)
    
    # 초기 중요도 기록
    feature_importance_history.append({
        'n_features': len(current_features),
        'importance': dict(zip(current_features, initial_imp))
    })

    while len(current_features) >= min_features:
        iter_start = time.time()
        n_feats = len(current_features)

        # 현재 피처로 학습
        X_tr = X_train_enc[current_features]
        X_vl = X_val_enc[current_features]

        model, X_tr_s, X_vl_s, auc, f1 = train_aitm_subset(
            X_tr, y_train, y_source_train, X_vl, y_val, verbose=False
        )

        print(f"  [N={n_feats}] AUC={auc:.4f}, F1={f1:.4f} ({time.time()-iter_start:.1f}s)")
        history.append({
            'n_features': n_feats,
            'auc': auc,
            'f1': f1,
            'features': copy.deepcopy(current_features)
        })

        if n_feats <= min_features:
            break

        # SHAP 중요도 재계산 (Recursive)
        # DataFrame으로 변환하여 전달 (컬럼명 매핑 위해)
        X_tr_df = pd.DataFrame(X_tr_s, columns=current_features)
        X_vl_df = pd.DataFrame(X_vl_s, columns=current_features)

        importances = calculate_shap_importance(model, X_tr_df, X_vl_df)

        # 중요도 하위 피처 제거
        imp_df = pd.DataFrame({'feature': current_features, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=True)

        # 현재 단계 중요도 기록 (제거 전 상태)
        feature_importance_history.append({
            'n_features': n_feats,
            'importance': dict(zip(current_features, importances))
        })
        
        # 제거할 피처 수
        n_remove = max(1, int(step)) if step >= 1 else max(1, int(n_feats * step))
        remove_cols = imp_df.head(n_remove)['feature'].tolist()

        # 남은 피처 업데이트
        current_features = [f for f in current_features if f not in remove_cols]

    return history, feature_importance_history


def plot_rfe_results(history, output_dir):
    """RFE 성능 그래프 시각화 및 최적점 선정"""
    df = pd.DataFrame(history)
    df = df.sort_values('n_features')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['n_features'], df['auc'], label='AUC', linewidth=2)
    # plt.plot(df['n_features'], df['f1'], marker='s', label='F1 (Max)')
    
    # 최적점 (AUC 기준)
    best_idx = df['auc'].idxmax()
    best_row = df.loc[best_idx]
    best_n = best_row['n_features']
    best_auc = best_row['auc']
    
    plt.axvline(x=best_n, color='r', linestyle='--', alpha=0.5, label=f'Optimal ({best_n})')
    plt.title(f"SHAP-RFE Feature Selection (Optimal: {best_n} features)", fontsize=14)
    plt.xlabel("Number of Features")
    plt.ylabel("Validation AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'shap_rfe_performance.png')
    plt.savefig(save_path)
    print(f"\nPerformance plot saved to {save_path}")
    plt.close()
    
    return best_row


def main():
    parser = argparse.ArgumentParser(description='SHAP-RFE Feature Selection')
    parser.add_argument('--step', type=float, default=1, help='제거할 피처 수 (1 이상: 개수, 1 미만: 비율)')
    parser.add_argument('--min_features', type=int, default=5, help='최소 피처 수')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SHAP-based Recursive Feature Elimination (RFE)")
    print("=" * 60)
    
    X_train, X_val, y_train, y_val, y_source_train = load_data()

    # RFE 5회 반복 실행 (Ensemble RFE)
    n_repeats = 5
    
    # 결과 집계용
    auc_stats = {}
    feature_importance_sum = Counter()
    
    print(f"\n[Multi-Run RFE] 총 {n_repeats}회 반복 실행 (Average AUC & Importance Voting)")
    
    output_dir = 'output/figures/shap'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n_repeats):
        seed = 38 + i
        print(f"\n>>> Run {i+1}/{n_repeats} (Seed={seed})")
        
        # RFE 실행
        history, imp_history = run_rfe(X_train, y_train, y_source_train, X_val, y_val,
                                       step=args.step, min_features=args.min_features, seed=seed)
        
        # AUC 집계
        for step_res in history:
            n = step_res['n_features']
            auc = step_res['auc']
            if n not in auc_stats:
                auc_stats[n] = []
            auc_stats[n].append(auc)
            
        # 중요도 집계 (초기 모델 기준 or 전체 평균)
        # 이번 Run에서 계산된 초기 중요도(모든 피처 포함)를 사용하여 합산
        # (imp_history[0]은 전체 피처에 대한 중요도)
        if imp_history:
            initial_imp = imp_history[0]['importance']
            for feat, val in initial_imp.items():
                feature_importance_sum[feat] += val

    print("=" * 60)
    print("Ensemble Feature Selection Result")
    print("=" * 60)

    # 1. 최적 피처 개수(Best K) 선정 - 평균 AUC 기준
    avg_auc_by_n = {}
    for n, aucs in auc_stats.items():
        avg_auc_by_n[n] = np.mean(aucs)
        
    # 평균 AUC가 가장 높은 N 찾기
    best_k = max(avg_auc_by_n, key=avg_auc_by_n.get)
    max_avg_auc = avg_auc_by_n[best_k]
    
    print(f"Optimal Number of Features (Besk K): {best_k}")
    print(f"Max Average AUC: {max_avg_auc:.4f}")
    
    # 2. 피처 랭킹 선정 - 중요도 합계 기준
    # 중요도 합계 내림차순 정렬
    sorted_features = feature_importance_sum.most_common()
    
    # 상위 K개 선정
    final_features = [f for f, _ in sorted_features[:best_k]]
    
    print(f"\n최종 선정된 피처 ({len(final_features)}개):")
    for i, f in enumerate(final_features):
        print(f"  {i+1}. {f} (Scale-Sum Imp: {feature_importance_sum[f]:.4f})")
        
    # 결과 그래프 (평균 AUC)
    lists = sorted(avg_auc_by_n.items()) 
    x, y = zip(*lists) 
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', label='Average AUC (5 Runs)')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal ({best_k})')
    plt.title(f"Ensemble RFE Performance (Best K={best_k})")
    plt.xlabel("Number of Features")
    plt.ylabel("Average Validation AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/figures/shap/shap_rfe_ensemble_avg.png')
    plt.close()

    # 피처 중요도 시각화 (Top 30)
    top_n = min(30, len(sorted_features))
    top_features = sorted_features[:top_n]
    
    feats = [f for f, val in top_features]
    vals = [val for f, val in top_features]
    
    plt.figure(figsize=(10, 8))
    plt.barh(feats[::-1], vals[::-1], color='skyblue')
    plt.xlabel("Total SHAP Importance (Sum over 5 Runs)")
    plt.title(f"Top {top_n} Feature Importance (Ensemble RFE)")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/figures/shap/shap_importance_ranking.png')
    plt.close()
    print(f"Saved feature importance plot to output/figures/shap/shap_importance_ranking.png")

    # 결과 저장
    result = {
        'selected_features': final_features,
        'n_features': best_k,
        'ensemble_runs': n_repeats,
        'best_avg_auc': float(max_avg_auc),
        'feature_importance_ranking': {f: val for f, val in sorted_features},
        'performance_by_n_features': {str(n): float(auc) for n, auc in avg_auc_by_n.items()}
    }
    
    json_path = 'output/shap_selected_features.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved ensemble optimal features to {json_path}")


if __name__ == '__main__':
    set_seed(MODEL_RANDOM_STATE)
    main()
