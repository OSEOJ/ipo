"""
9_benchmark.py - 모델 학습 및 벤치마크 (통합)

기능:
1. 7개 모델 전체 벤치마크 (기본)
2. 단일 모델 지정 학습 (--model 옵션)
3. SHAP 선정 피처 자동 적용 (output/shap_selected_features.json 존재 시)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
import subprocess
import warnings
warnings.filterwarnings('ignore')
import torch

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
)

import koreanize_matplotlib  # 한국어 폰트 설정 (side-effect import)
plt.rcParams['axes.unicode_minus'] = False


# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_RANDOM_STATE, MTL_MANUAL_SOURCE_TASKS,
)
from src.utils import set_seed, evaluate_with_threshold
from src.models.factory import create_model
from src.models.common import calc_permutation_importance
from src.models.conflict_analysis import plot_all_conflict_analysis, print_conflict_summary
from src.data_pipeline import IPODataPipeline


# ============================================================
# 벤치마크 대상 모델 정의
# ============================================================
ALL_MODELS = {
    'AITM': {'type': 'dl_mtl', 'desc': 'Adaptive Information Transfer MTL (병렬)'},
    'AITM_Seq': {'type': 'dl_mtl', 'desc': 'Original Sequential AITM (Baseline)'},
    'MMoE': {'type': 'dl_mtl', 'desc': 'Multi-gate Mixture-of-Experts'},
    'PLE':  {'type': 'dl_mtl', 'desc': 'Progressive Layered Extraction'},
    'SingleTask': {'type': 'dl_mtl', 'desc': 'Task별 독립 MLP'},
    'XGBoost': {'type': 'ml', 'desc': 'Gradient Boosting'},
    'CatBoost': {'type': 'ml', 'desc': 'Ordered Boosting'},
    'LogisticRegression': {'type': 'ml', 'desc': 'Linear Baseline'},
}


def load_data(apply_shap=True):
    """전처리된 데이터 로드 (IPODataPipeline 사용)"""
    pipeline = IPODataPipeline()
    X, y, y_source, listing_dates, _stock_codes = pipeline.load()

    # SHAP 피처 로드
    shap_features = None
    if apply_shap:
        shap_features = pipeline.load_shap_features()

    return pipeline, X, y, y_source, listing_dates, shap_features


def evaluate_model(model, X_test, y_test):
    """모델 평가 (Threshold Moving)"""
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test).astype(float)

    results = evaluate_with_threshold(y_test, y_pred_proba)

    return results, y_pred_proba


def compute_feature_importance(model, model_name, X_test, y_test):
    """Feature Importance 계산 (값만 반환, 시각화는 fold 종합 후 별도 수행)"""
    # 1. Tree-based Models (XGBoost, CatBoost)
    if getattr(model, 'feature_importances_', None) is not None:
        return model.feature_importances_

    # 2. Linear Models (Logistic)
    if hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])

    # 3. Deep Learning Models (Permutation Importance)
    if hasattr(model, 'predict_proba') and hasattr(model, 'model'):
        device = getattr(model, 'device', None)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"  [Calc] Calculating Permutation Importance for {model_name}...")
        return calc_permutation_importance(
            model.model, X_test.values, y_test.values, device
        )

    return None


def save_mean_roc(fold_data, model_name, save_dir='output/figures/benchmark'):
    """Mean ROC Curve + std band (K-Fold 종합)"""
    os.makedirs(save_dir, exist_ok=True)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    plt.figure(figsize=(8, 6))

    for i, fd in enumerate(fold_data):
        fpr, tpr, _ = roc_curve(fd['y_true'], fd['y_pred_proba'])
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, alpha=0.3, linewidth=1,
                 label=f'Fold {i+1} (AUC={auc_score:.4f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    roc_df = pd.DataFrame({
        'FPR': mean_fpr,
        'Mean_TPR': mean_tpr,
        'Std_TPR': std_tpr
    })
    csv_path = os.path.join(save_dir, f'roc_{model_name}.csv')
    roc_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  -> Saved Mean ROC data: {csv_path}")

    plt.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
             label=f'Mean ROC (AUC={mean_auc:.4f} +/- {std_auc:.4f})')
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='b', alpha=0.15,
                     label='+/- 1 std')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Mean ROC Curve ({len(fold_data)}-Fold)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    roc_path = os.path.join(save_dir, f'roc_{model_name}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved Mean ROC: {roc_path}")


def save_aggregate_cm(fold_data, model_name, save_dir='output/figures/benchmark'):
    """Aggregate Confusion Matrix (전체 fold 예측 합산)"""
    os.makedirs(save_dir, exist_ok=True)

    all_y_true = np.concatenate([fd['y_true'] for fd in fold_data])
    all_y_pred = np.concatenate([fd['y_pred'] for fd in fold_data])

    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Negative', 'Positive'])

    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.grid(False) # Seaborn 테마로 인해 생긴 격자선 강제 제거
    plt.title(f'{model_name} Confusion Matrix (Aggregate {len(fold_data)}-Fold)')

    cm_path = os.path.join(save_dir, f'cm_{model_name}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved Aggregate CM: {cm_path}")


def save_mean_feature_importance(fold_data, model_name, save_dir='output/figures/benchmark'):
    """Mean Feature Importance + std error bar (K-Fold 종합)"""
    os.makedirs(save_dir, exist_ok=True)

    valid = [fd for fd in fold_data if fd['importances'] is not None]
    if not valid:
        print(f"  [Info] No feature importance data for {model_name}")
        return

    feature_names = valid[0]['feature_names']
    imp_matrix = np.vstack([fd['importances'] for fd in valid])
    mean_imp = np.mean(imp_matrix, axis=0)
    std_imp = np.std(imp_matrix, axis=0)

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': mean_imp,
        'Std_Importance': std_imp
    }).sort_values('Mean_Importance', ascending=False)
    csv_path = os.path.join(save_dir, f'fi_{model_name}.csv')
    fi_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  -> Saved Mean FI data: {csv_path}")

    # Top 20
    indices = np.argsort(mean_imp)[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_mean = mean_imp[indices]
    top_std = std_imp[indices]

    plt.figure(figsize=(10, 8))
    y_pos = range(len(indices))
    plt.barh(y_pos, top_mean[::-1], xerr=top_std[::-1], align='center',
             capsize=3, color='steelblue', alpha=0.8)
    plt.yticks(y_pos, top_features[::-1])
    plt.xlabel('Importance')
    plt.title(f'{model_name} Top 20 Feature Importances '
              f'(Mean +/- Std, {len(valid)}-Fold)')
    
    plt.tight_layout()

    fi_path = os.path.join(save_dir, f'fi_{model_name}.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved Mean FI: {fi_path}")




def compute_mcnemar_tests(model_preds, seed=None):
    """모델 간 McNemar's test (pairwise, K-Fold 통합 예측 기반)

    Args:
        model_preds: {model_name: {'y_true': array, 'y_pred': array}, ...}
    """
    from scipy.stats import chi2

    model_names = list(model_preds.keys())
    n = len(model_names)

    if n < 2:
        print("\n[McNemar's Test] 비교할 모델이 2개 이상 필요합니다.")
        return

    print("\n" + "=" * 70)
    print("  McNemar's Test (Pairwise, Edwards 연속 수정 적용)")
    print("  ns: p>=0.05  *: p<0.05  **: p<0.01")
    print("  b=A맞고B틀림, c=A틀리고B맞음")
    print("=" * 70)

    pair_results = {}  # (name_a, name_b) -> (b, c, chi2, p, sig)

    for i in range(n):
        for j in range(i + 1, n):
            name_a, name_b = model_names[i], model_names[j]

            y_true_a = np.array(model_preds[name_a]['y_true'])
            y_pred_a = np.array(model_preds[name_a]['y_pred'])
            y_true_b = np.array(model_preds[name_b]['y_true'])
            y_pred_b = np.array(model_preds[name_b]['y_pred'])

            if len(y_true_a) != len(y_true_b) or not np.array_equal(y_true_a, y_true_b):
                print(f"  [Warning] {name_a} vs {name_b}: y_true 불일치, 건너뜀")
                continue

            correct_a = (y_pred_a == y_true_a)
            correct_b = (y_pred_b == y_true_b)
            b = int(np.sum(correct_a & ~correct_b))  # A맞고 B틀림
            c = int(np.sum(~correct_a & correct_b))  # A틀리고 B맞음

            if b + c == 0:
                chi2_stat, p_val = 0.0, 1.0
            else:
                chi2_stat = float((abs(b - c) - 1) ** 2 / (b + c))
                p_val = float(1 - chi2.cdf(chi2_stat, df=1))

            sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
            pair_results[(name_a, name_b)] = (b, c, chi2_stat, p_val, sig)

            print(f"  {name_a:20s} vs {name_b:20s}: "
                  f"b={b:4d}, c={c:4d}, chi2={chi2_stat:7.3f}, p={p_val:.4f} [{sig}]")

    # p-value 행렬
    col_w = 11
    print(f"\n  p-value 행렬:")
    header = f"  {'':22s}" + "".join(f"{m[:col_w]:>{col_w}s}" for m in model_names)
    print(header)
    for i, name_a in enumerate(model_names):
        row = f"  {name_a:22s}"
        for j, name_b in enumerate(model_names):
            if i == j:
                row += f"{'---':>{col_w}s}"
            else:
                key = (name_a, name_b) if (name_a, name_b) in pair_results else (name_b, name_a)
                if key in pair_results:
                    p, sig = pair_results[key][3], pair_results[key][4]
                    cell = f"{p:.4f}{sig}"
                else:
                    cell = "N/A"
                row += f"{cell:>{col_w}s}"
        print(row)
    print()

    # CSV 저장
    save_dir = 'output/mcnemar'
    os.makedirs(save_dir, exist_ok=True)
    suffix = f'_seed{seed}' if seed is not None else ''

    # 1) pairwise 전체 표
    rows = []
    for (na, nb), (b, c, chi2_s, p, sig) in pair_results.items():
        rows.append({'Model_A': na, 'Model_B': nb, 'b': b, 'c': c,
                     'chi2': round(chi2_s, 4), 'p_value': round(p, 4), 'sig': sig})
    pairwise_df = pd.DataFrame(rows)
    pairwise_path = os.path.join(save_dir, f'pairwise{suffix}.csv')
    pairwise_df.to_csv(pairwise_path, index=False, encoding='utf-8-sig')
    print(f"  -> Saved pairwise McNemar: {pairwise_path}")

    # 2) AITM 기준 요약 (AITM이 있을 때만)
    ref = 'AITM'
    if ref in model_names:
        aitm_rows = []
        for other in model_names:
            if other == ref:
                continue
            key = (ref, other) if (ref, other) in pair_results else (other, ref)
            if key not in pair_results:
                continue
            b, c, chi2_s, p, sig = pair_results[key]
            # 방향 보정: key[0]이 ref가 아니면 b/c 교환
            if key[0] != ref:
                b, c = c, b
            formatted = f"{chi2_s:.3f}{sig} ({p:.3f})"
            aitm_rows.append({'Model': other, 'b': b, 'c': c,
                               'chi2': round(chi2_s, 4), 'p_value': round(p, 4),
                               'sig': sig, 'formatted': formatted})
        if aitm_rows:
            aitm_df = pd.DataFrame(aitm_rows)
            aitm_path = os.path.join(save_dir, f'aitm_vs_others{suffix}.csv')
            aitm_df.to_csv(aitm_path, index=False, encoding='utf-8-sig')

            print(f"\n  [AITM vs 타 모델 McNemar's test]")
            for r in aitm_rows:
                print(f"  {r['Model']:20s}: {r['formatted']}")
            print(f"  -> Saved AITM summary: {aitm_path}")


def train_and_evaluate(model_name, X_train, y_train, y_source_train, X_test, y_test, source_days, fold_idx=None):
    """단일 모델 학습 및 평가"""
    info = ALL_MODELS.get(model_name, {'type': 'custom', 'desc': 'Custom Model'})
    print(f"\n{'='*50}")
    print(f"  Training: {model_name} ({info['desc']})")
    print(f"{'='*50}")

    # 모델 파라미터: config.py의 MLP_PARAMS를 사용
    model_params = {}

    X_fit, y_fit = X_train, y_train
    y_source_fit = y_source_train

    # y_dict (MTL용) — prepare_labels 대신 직접 구성 (inner split된 데이터이므로)
    y_train_dict = {'target': y_fit}
    for i, (col, val) in enumerate(y_source_fit.items()):
        y_train_dict[f'source_{i}'] = val

    start_time = time.time()

    # 모델 생성 (AITM인 경우 analyze_conflict=True)
    if model_name == 'AITM':
        model_params['analyze_conflict'] = True
    model = create_model(model_name, source_days=source_days, verbose=True, **model_params)

    # 학습
    if info['type'] == 'dl_mtl' and y_source_train:
        model.fit(X_fit, y_train_dict)
    else:
        model.fit(X_fit, y_fit)

    elapsed = time.time() - start_time

    # 평가 (Validation)
    results, y_pred_proba = evaluate_model(model, X_test, y_test)
    results['Model'] = model_name
    results['Type'] = info['type']
    results['Time(s)'] = round(elapsed, 1)

    # 평가 (Train) - 과적합 확인용
    train_results, _ = evaluate_model(model, X_fit, y_fit)

    print(f"  -> [Train] Acc={train_results['Accuracy']:.4f}, F1={train_results['F1']:.4f}, "
          f"AUC={train_results['AUC']:.4f}")
    print(f"  -> [Val]   Acc={results['Accuracy']:.4f}, F1={results['F1']:.4f}, "
          f"AUC={results['AUC']:.4f}, Time={elapsed:.1f}s")

    # Feature Importance 계산 (시각화는 fold 종합 후)
    importances = compute_feature_importance(model, model_name, X_test, y_test)

    # AITM: Task Conflict Analysis 시각화
    if model_name == 'AITM' and hasattr(model, 'gradient_cosine_history'):
        print_conflict_summary(model)
        suffix = f'_fold{fold_idx}' if fold_idx is not None else ''
        plot_all_conflict_analysis(model, save_path='output/figures/conflict', suffix=suffix)

    return results, train_results, model, y_pred_proba, importances


def main(model=None, no_shap=False, seed=None, n_seeds=1, seed_step=1, ckpt_suffix='', save_path='', n_folds=3):
    # 키워드 인자가 명시적으로 전달된 경우 argparse 건너뛰기
    if model is not None:
        class Args: pass
        args = Args()
        args.model = model
        args.no_shap = no_shap
        args.seed = seed if seed is not None else MODEL_RANDOM_STATE
        args.n_seeds = n_seeds
        args.seed_step = seed_step
        args.ckpt_suffix = ckpt_suffix
        args.save_path = save_path
        args.n_folds = n_folds
    else:
        parser = argparse.ArgumentParser(description='Integrated Benchmark/Train Script')
        parser.add_argument('--model', type=str, default='all', 
                            help='실행할 모델 이름 (all, aitm, xgboost, ...)')
        parser.add_argument('--no-shap', action='store_true', help='SHAP 피처 선택 비활성화')
        parser.add_argument('--n-folds', type=int, default=3, help='K-Fold 수 (기본 3)')
        parser.add_argument('--seed', type=int, default=MODEL_RANDOM_STATE, help='랜덤 시드')
        parser.add_argument('--n-seeds', type=int, default=1, help='반복 실행할 seed 개수 (기본 1)')
        parser.add_argument('--seed-step', type=int, default=1, help='seed 증가 간격 (기본 1)')
        parser.add_argument('--ckpt-suffix', type=str, default='', help='AITM 체크포인트 파일명 suffix (예: _seed43)')
        parser.add_argument('--save-path', type=str, default='', help='결과 CSV 저장 경로 (내부/외부 집계용)')
        args = parser.parse_args()

    # ------------------------------------------------------------
    # Multi-Seed 모드: seed별 단일 실행을 재호출하고 결과를 집계
    # ------------------------------------------------------------
    if args.n_seeds > 1:
        seeds = [args.seed + i * args.seed_step for i in range(args.n_seeds)]
        bench_dir = 'output/benchmark'
        mc_dir = 'output/mcnemar'
        os.makedirs(bench_dir, exist_ok=True)
        os.makedirs(mc_dir, exist_ok=True)

        per_seed_paths = []
        for seed in seeds:
            per_seed_path = os.path.join(bench_dir, f'results_seed{seed}.csv')
            per_seed_paths.append(per_seed_path)

            seed_ckpt_suffix = f"{args.ckpt_suffix}_seed{seed}" if args.ckpt_suffix else f"_seed{seed}"

            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                '--model', args.model,
                '--n-folds', str(args.n_folds),
                '--seed', str(seed),
                '--n-seeds', '1',
                '--seed-step', str(args.seed_step),
                '--ckpt-suffix', seed_ckpt_suffix,
                '--save-path', per_seed_path,
            ]
            if args.no_shap:
                cmd.append('--no-shap')

            print(f"\n[Multi-Seed] Running seed={seed}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # seed별 결과 결합
        frames = []
        for path in per_seed_paths:
            if os.path.exists(path):
                df_seed = pd.read_csv(path)
                frames.append(df_seed)

        if not frames:
            print("[Multi-Seed] No result CSV found. Please check per-seed runs.")
            return

        all_df = pd.concat(frames, ignore_index=True)

        # 수치 컬럼 평균/표준편차 집계
        metric_cols = [c for c in [
            'Train_Accuracy', 'Train_F1', 'Train_AUC', 'Train_Precision', 'Train_Recall',
            'Accuracy', 'F1', 'AUC', 'Precision', 'Recall', 'Time(s)',
        ] if c in all_df.columns]
        agg = all_df.groupby(['Model'], as_index=False)[metric_cols].agg(['mean', 'std'])
        agg.columns = ['Model'] + [f"{col}_{stat}" for col, stat in agg.columns.tolist()[1:]]
        if 'F1_mean' in agg.columns:
            agg = agg.sort_values('F1_mean', ascending=False)

        raw_path = os.path.join(bench_dir, 'results_multiseed_raw.csv')
        agg_path = args.save_path if args.save_path else os.path.join(bench_dir, 'results_multiseed.csv')
        all_df.to_csv(raw_path, index=False, encoding='utf-8-sig')
        agg.to_csv(agg_path, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 60)
        print(f"  MULTI-SEED RESULTS SUMMARY (n_seeds={args.n_seeds})")
        print("=" * 60)
        print(agg.to_string(index=False))
        print(f"\n-> Saved raw per-seed concat: {raw_path}")
        print(f"-> Saved multi-seed aggregate: {agg_path}")

        # McNemar pairwise 집계
        mc_frames = []
        for seed in seeds:
            mc_path = os.path.join(mc_dir, f'pairwise_seed{seed}.csv')
            if os.path.exists(mc_path):
                df_mc = pd.read_csv(mc_path)
                df_mc['Seed'] = seed
                mc_frames.append(df_mc)
        if mc_frames:
            mc_all = pd.concat(mc_frames, ignore_index=True)
            mc_agg = mc_all.groupby(['Model_A', 'Model_B'], as_index=False)[['chi2', 'p_value']].agg(['mean', 'std'])
            mc_agg.columns = ['Model_A', 'Model_B'] + [
                f"{col}_{stat}" for col, stat in mc_agg.columns.tolist()[2:]
            ]
            mc_agg['sig'] = mc_agg['p_value_mean'].apply(
                lambda p: '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
            )
            mc_agg['formatted'] = mc_agg.apply(
                lambda r: f"{r['chi2_mean']:.3f}{r['sig']} ({r['p_value_mean']:.3f})", axis=1
            )
            mc_all.to_csv(os.path.join(mc_dir, 'pairwise_multiseed_raw.csv'),
                          index=False, encoding='utf-8-sig')
            mc_agg.to_csv(os.path.join(mc_dir, 'pairwise_multiseed.csv'),
                          index=False, encoding='utf-8-sig')
            print(f"-> Saved McNemar pairwise multiseed: {mc_dir}/pairwise_multiseed.csv")

        # McNemar AITM vs others 집계
        aitm_frames = []
        for seed in seeds:
            aitm_path = os.path.join(mc_dir, f'aitm_vs_others_seed{seed}.csv')
            if os.path.exists(aitm_path):
                df_a = pd.read_csv(aitm_path)
                df_a['Seed'] = seed
                aitm_frames.append(df_a)
        if aitm_frames:
            aitm_all = pd.concat(aitm_frames, ignore_index=True)
            aitm_agg = aitm_all.groupby(['Model'], as_index=False)[['chi2', 'p_value']].agg(['mean', 'std'])
            aitm_agg.columns = ['Model'] + [
                f"{col}_{stat}" for col, stat in aitm_agg.columns.tolist()[1:]
            ]
            aitm_agg['sig'] = aitm_agg['p_value_mean'].apply(
                lambda p: '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
            )
            aitm_agg['formatted'] = aitm_agg.apply(
                lambda r: f"{r['chi2_mean']:.3f}{r['sig']} ({r['p_value_mean']:.3f})", axis=1
            )
            aitm_all.to_csv(os.path.join(mc_dir, 'aitm_vs_others_multiseed_raw.csv'),
                            index=False, encoding='utf-8-sig')
            aitm_agg.to_csv(os.path.join(mc_dir, 'aitm_vs_others_multiseed.csv'),
                            index=False, encoding='utf-8-sig')
            print(f"\n  [AITM vs 타 모델 McNemar's test (Multi-Seed 평균)]")
            for _, r in aitm_agg.iterrows():
                print(f"  {r['Model']:20s}: {r['formatted']}")
            print(f"-> Saved AITM multiseed: {mc_dir}/aitm_vs_others_multiseed.csv")

        return

    N_FOLDS = args.n_folds

    print("=" * 60)
    print(f"  IPO Prediction - {args.model.upper()} ({N_FOLDS}-Fold TimeSeriesSplit)")
    print("=" * 60)

    set_seed(args.seed)

    # 1. 데이터 로드
    print("\n[Step 1] Loading data...")
    pipeline, X, y, y_source, listing_dates, shap_features = load_data(apply_shap=not args.no_shap)

    # 2. Train/Test Split (시계열) — Pipeline 사용
    X_train_raw, X_test_raw, y_train_combined, y_test_combined = pipeline.get_train_test()

    # Source Days
    if MTL_MANUAL_SOURCE_TASKS:
        source_days = list(MTL_MANUAL_SOURCE_TASKS)
    else:
        source_days = []

    # 3. 대상 모델 결정
    target_models = []
    if args.model == 'all':
        target_models = list(ALL_MODELS.keys())
    else:
        for m in ALL_MODELS.keys():
            if m.lower() == args.model.lower():
                target_models = [m]
                break
        if not target_models:
            print(f"Error: Unknown model '{args.model}'")
            return

    # 4. K-Fold 학습 및 평가
    results_list = []
    model_combined_preds = {}  # McNemar's test용 모델별 통합 예측

    for model_name in target_models:
        info = ALL_MODELS.get(model_name, {'type': 'custom', 'desc': 'Custom Model'})
        print(f"\n{'='*50}")
        print(f"  {model_name} ({info['desc']}) - {N_FOLDS}-Fold")
        print(f"{'='*50}")

        fold_results = []
        train_fold_results = []
        fold_vis_data = []
        combined_y_true = []
        combined_y_pred = []
        total_start = time.time()

        for fold_idx, X_tr_raw, X_val_raw, y_tr_combined, y_val_combined in pipeline.get_cv_folds(
            X_train_raw, y_train_combined, n_folds=N_FOLDS
        ):
            # Fold별 인코딩/스케일링 (데이터 누수 방지)
            X_tr, X_val = pipeline.process(X_tr_raw, X_val_raw, shap_features=shap_features, verbose=False)

            y_tr = y_tr_combined['Y']
            y_val = y_val_combined['Y']
            y_source_fold = {col: y_tr_combined[col] for col in y_source.keys() if col in y_tr_combined.columns}

            # 학습 & 평가
            res, train_res, _model, y_pred_proba, importances = train_and_evaluate(
                model_name, X_tr, y_tr, y_source_fold, X_val, y_val, source_days,
                fold_idx=fold_idx,
            )
            if res:
                fold_results.append(res)
                train_fold_results.append(train_res)
                y_pred = (np.asarray(y_pred_proba) >= res['Threshold']).astype(int)
                y_true_arr = y_val.values if hasattr(y_val, 'values') else np.asarray(y_val)

                combined_y_true.extend(y_true_arr.tolist())
                combined_y_pred.extend(y_pred.tolist())

                fold_vis_data.append({
                    'y_true': y_true_arr,
                    'y_pred_proba': np.asarray(y_pred_proba),
                    'y_pred': y_pred,
                    'importances': importances,
                    'feature_names': X_val.columns.tolist(),
                })

        total_elapsed = time.time() - total_start

        # McNemar's test용 통합 예측 저장
        if combined_y_true:
            model_combined_preds[model_name] = {
                'y_true': np.array(combined_y_true),
                'y_pred': np.array(combined_y_pred),
            }

        # Fold 종합 시각화
        if fold_vis_data:
            save_mean_roc(fold_vis_data, model_name)
            save_aggregate_cm(fold_vis_data, model_name)
            save_mean_feature_importance(fold_vis_data, model_name)

        # Fold 결과 평균
        if fold_results:
            avg_result = {
                'Model': model_name,
                'Type': info['type'],
                'Train_Accuracy': np.mean([r['Accuracy'] for r in train_fold_results]),
                'Train_F1': np.mean([r['F1'] for r in train_fold_results]),
                'Train_AUC': np.mean([r['AUC'] for r in train_fold_results]),
                'Train_Precision': np.mean([r['Precision'] for r in train_fold_results]),
                'Train_Recall': np.mean([r['Recall'] for r in train_fold_results]),
                'Accuracy': np.mean([r['Accuracy'] for r in fold_results]),
                'F1': np.mean([r['F1'] for r in fold_results]),
                'AUC': np.mean([r['AUC'] for r in fold_results]),
                'Precision': np.mean([r['Precision'] for r in fold_results]),
                'Recall': np.mean([r['Recall'] for r in fold_results]),
                'Time(s)': round(total_elapsed, 1),
            }
            # Fold별 상세 출력
            print(f"\n  [{model_name}] Fold Results:")
            print(f"  {'Fold':>6s}  {'Train Acc':>9s}  {'Train F1':>8s}  {'Train AUC':>9s}"
                  f"  {'Val Acc':>7s}  {'Val F1':>6s}  {'Val AUC':>7s}")
            for i, (r, tr) in enumerate(zip(fold_results, train_fold_results)):
                print(f"  {i+1:>6d}  {tr['Accuracy']:>9.4f}  {tr['F1']:>8.4f}  {tr['AUC']:>9.4f}"
                      f"  {r['Accuracy']:>7.4f}  {r['F1']:>6.4f}  {r['AUC']:>7.4f}")
            print(f"  {'Mean':>6s}  {avg_result['Train_Accuracy']:>9.4f}  {avg_result['Train_F1']:>8.4f}"
                  f"  {avg_result['Train_AUC']:>9.4f}"
                  f"  {avg_result['Accuracy']:>7.4f}  {avg_result['F1']:>6.4f}  {avg_result['AUC']:>7.4f}")

            results_list.append(avg_result)

    # 5. AITM 체크포인트 저장 (Attention 분석용)
    if 'AITM' in target_models:
        print(f"\n{'='*50}")
        print("  [AITM] Saving checkpoint for attention analysis...")
        print(f"{'='*50}")

        # 전체 Train 데이터로 최종 모델 학습
        X_tr_full, _ = pipeline.process(
            X_train_raw, X_test_raw, shap_features=shap_features, verbose=False
        )
        y_tr_full = y_train_combined['Y']
        y_source_full = {
            col: y_train_combined[col]
            for col in y_source.keys() if col in y_train_combined.columns
        }
        y_train_dict_full = {'target': y_tr_full}
        for i, (col, val) in enumerate(y_source_full.items()):
            y_train_dict_full[f'source_{i}'] = val

        final_model = create_model('AITM', source_days=source_days, verbose=True)
        final_model.fit(X_tr_full, y_train_dict_full)

        ckpt_filename = f"aitm_checkpoint{args.ckpt_suffix}.pt"
        ckpt_dir = 'output/checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
        final_model.save_checkpoint(ckpt_path, X_tr_full.columns.tolist())

    # 6. McNemar's Test (모델 간 통계적 유의성 비교)
    if len(model_combined_preds) >= 2:
        compute_mcnemar_tests(model_combined_preds, seed=args.seed)

    # 7. 결과 출력
    if not results_list:
        return

    results_df = pd.DataFrame(results_list)
    results_df['Seed'] = args.seed
    results_df = results_df.sort_values('F1', ascending=False)

    train_cols = ['Model', 'Train_Accuracy', 'Train_F1', 'Train_AUC', 'Time(s)']
    val_cols   = ['Model', 'Accuracy', 'F1', 'AUC', 'Time(s)']
    available_train_cols = [c for c in train_cols if c in results_df.columns]

    print("\n" + "=" * 60)
    print(f"  RESULTS SUMMARY ({N_FOLDS}-Fold Mean) — Train Set")
    print("=" * 60)
    if available_train_cols:
        print(results_df[available_train_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"  RESULTS SUMMARY ({N_FOLDS}-Fold Mean) — Validation Set")
    print("=" * 60)
    print(results_df[val_cols].to_string(index=False))

    # 결과 저장 (save-path 우선, 없으면 기존 동작 유지)
    csv_path = None
    if args.save_path:
        csv_path = args.save_path
    elif args.model == 'all':
        output_dir = 'output/benchmark'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'results.csv')

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n-> Saved to {csv_path}")

if __name__ == "__main__":
    main()
