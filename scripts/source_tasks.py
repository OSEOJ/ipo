"""
Source Task 선정 통합 스크립트

mode:
  optimize — Optuna로 Source Task 선정 파라미터 최적화 (기존 4_source_optimize.py)
  select   — 최적 파라미터로 Source Task 선정 + 시각화 (기존 5_source_selection.py)
  auto     — optimize → select 순차 실행

실행:
  python scripts/source_tasks.py --mode auto
  python scripts/source_tasks.py --mode optimize --trials 50
  python scripts/source_tasks.py --mode select --manual-select 1 3 68
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PREDICTION_HORIZON, MLP_PARAMS, SCALER
from src.source_selection import (
    load_bhar_trend,
    compute_correlations,
    greedy_select_sources,
    compute_pairwise_r_squared,
)

# ============================================================
# optimize 모드 (기존 4_source_optimize.py)
# ============================================================

# 전역 변수
BHAR_PIVOT = None
PIPELINE = None
X_TRAIN_ALL = None
Y_TRAIN_COMBINED = None
CODES_TRAIN = None


def _load_global_data():
    """데이터를 한 번만 로드하여 전역 변수에 저장"""
    global BHAR_PIVOT, PIPELINE, X_TRAIN_ALL, Y_TRAIN_COMBINED, CODES_TRAIN

    print("Loading global data for Optuna...")

    if os.path.exists('output/ipo_bhar_trend_160d.csv'):
        BHAR_PIVOT = load_bhar_trend('output/ipo_bhar_trend_160d.csv')
        # T=150까지만 사용 (사용자 요청)
        BHAR_PIVOT = BHAR_PIVOT.loc[:, BHAR_PIVOT.columns <= 150]
        print(f"Filtered BHAR Trend Data: T <= 150 (Total {len(BHAR_PIVOT.columns)} days)")
    else:
        raise FileNotFoundError("output/ipo_bhar_trend_160d.csv not found.")

    from src.data_pipeline import IPODataPipeline
    PIPELINE = IPODataPipeline()
    PIPELINE.load()

    X_train, _X_test, y_train, _y_test, codes_train, _codes_test = PIPELINE.get_train_test_with_codes()

    X_TRAIN_ALL = X_train
    Y_TRAIN_COMBINED = y_train
    CODES_TRAIN = codes_train

    print(f"Data Loaded: {len(X_TRAIN_ALL)} train samples (Test 제외)")


def _objective(trial):
    """Optuna Objective Function"""
    import optuna
    from sklearn.metrics import f1_score

    from src.models.aitm import AITMClassifier

    global BHAR_PIVOT, PIPELINE, X_TRAIN_ALL, Y_TRAIN_COMBINED, CODES_TRAIN

    corr_min = trial.suggest_float('corr_min', 0.1, 0.5)
    corr_max = trial.suggest_float('corr_max', 0.5, 0.9)
    max_pairwise_r2 = trial.suggest_float('max_pairwise_r2', 0.2, 0.8)
    max_sources = trial.suggest_int('max_sources', 2, 4)

    if corr_min >= corr_max:
        raise optuna.exceptions.TrialPruned()

    correlations = compute_correlations(BHAR_PIVOT, PREDICTION_HORIZON)
    selected = greedy_select_sources(
        correlations, BHAR_PIVOT, PREDICTION_HORIZON,
        corr_min=corr_min, corr_max=corr_max,
        max_pairwise_r2=max_pairwise_r2, max_sources=max_sources,
    )

    if not selected:
        return 0.0

    selected_days = [s['day'] for s in selected]

    # y_source 구성
    y_source_dict = {}
    y_target = Y_TRAIN_COMBINED['Y']

    for day in selected_days:
        if day in BHAR_PIVOT.columns:
            bhar_series = BHAR_PIVOT[day]
            mapped_bhar = CODES_TRAIN.map(bhar_series)
            y_task = (mapped_bhar >= 0).astype(y_target.dtype)

            mask_nan = mapped_bhar.isna()
            if mask_nan.any():
                y_task.loc[mask_nan] = y_target.loc[mask_nan]

            y_source_dict[f'Y_T{day}'] = y_task

    if not y_source_dict:
        return 0.0

    # 3-Fold CV
    scores = []

    for fold_idx, X_tr_raw, X_val_raw, y_tr_combined, y_val_combined in PIPELINE.get_cv_folds(
        X_TRAIN_ALL, Y_TRAIN_COMBINED, n_folds=3
    ):
        X_train, X_val = PIPELINE.process(X_tr_raw, X_val_raw, verbose=False)

        y_train = y_tr_combined['Y']
        y_val = y_val_combined['Y']

        tr_size = len(X_tr_raw)
        y_source_train = {}
        for k, v in y_source_dict.items():
            y_source_train[k] = v.iloc[:tr_size].reset_index(drop=True)



        y_train_mtl = {'target': y_train}
        for i, (k, v) in enumerate(y_source_train.items()):
            y_train_mtl[f'source_{i}'] = v

        y_valid_mtl = {'target': y_val}

        model = AITMClassifier(
            bottom_mlp_dims=MLP_PARAMS.get('hidden_dims', [128, 64]),
            tower_mlp_dims=MLP_PARAMS.get('tower_dims', [64]),
            dropout=MLP_PARAMS.get('dropout', 0.2),
            learning_rate=MLP_PARAMS.get('learning_rate', 1e-3),
            batch_size=MLP_PARAMS.get('batch_size', 256),
            epochs=50,
            source_days=selected_days,
            verbose=False,
        )

        try:
            model.fit(X_train, y_train_mtl, X_valid=X_val, y_valid_dict=y_valid_mtl)

            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            scores.append(f1)
        except Exception as e:
            print(f"  [Trial {trial.number}, Fold {fold_idx}] Error: {e}")
            continue

        trial.report(np.mean(scores), fold_idx)
        if trial.should_prune():
            import optuna
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores) if scores else 0.0


def run_optimize(n_trials=30):
    """Optuna 기반 Source Task 파라미터 최적화"""
    import optuna

    _load_global_data()

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(_objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("Optimization Result")
    print("=" * 60)
    print(f"Best F1 Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 최적 파라미터로 선택된 Source Tasks 확인
    print("\nSelected Source Tasks with Best Params:")

    pivot = load_bhar_trend('output/ipo_bhar_trend_160d.csv')
    correlations = compute_correlations(pivot, PREDICTION_HORIZON)
    greedy_select_sources(
        correlations, pivot, PREDICTION_HORIZON,
        corr_min=study.best_params['corr_min'],
        corr_max=study.best_params['corr_max'],
        max_pairwise_r2=study.best_params['max_pairwise_r2'],
        max_sources=study.best_params['max_sources'],
        verbose=True
    )

    # 결과 저장
    best_params_path = 'output/best_source_params.json'
    os.makedirs('output', exist_ok=True)
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n[Success] 최적 파라미터가 {best_params_path}에 저장되었습니다.")
    print("-> run_select() 실행 시 자동 적용됩니다.")


# ============================================================
# select 모드 (기존 5_source_selection.py)
# ============================================================

def _plot_correlation_heatmap(correlations, target_day, selected_days, output_dir, corr_min=None, corr_max=None):
    """상관관계 히트맵 시각화"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import koreanize_matplotlib
    except ImportError:
        pass

    os.makedirs(output_dir, exist_ok=True)

    days = sorted(correlations.keys())
    corrs = [correlations[d]['correlation'] for d in days]
    r2s = [correlations[d]['r_squared'] for d in days]
    igs = [correlations[d]['information_gain'] for d in days]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax1 = axes[0]
    ax1.plot(days, corrs, 'b-', alpha=0.7, linewidth=1)
    
    if corr_min is not None:
        ax1.axhline(y=corr_min, color='g', linestyle='--', alpha=0.5, label='하한')
    if corr_max is not None:
        ax1.axhline(y=corr_max, color='r', linestyle='--', alpha=0.5, label='상한')
        
    ax1.axvline(x=target_day, color='orange', linestyle='-', alpha=0.8, label=f'Target (T={target_day})')

    for d in selected_days:
        ax1.axvline(x=d, color='purple', linestyle=':', alpha=0.6)
        # 점(scatter) 제거 (사용자 요청)
        # ax1.scatter([d], [correlations[d]['correlation']], color='red', s=100, zorder=5)

    ax1.set_ylabel('Pearson 상관계수')
    ax1.set_title(f'Target Task (T={target_day}) 대비 각 시점의 상관관계')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(days, r2s, alpha=0.3, color='blue')
    ax2.plot(days, r2s, 'b-', linewidth=1)
    ax2.axvline(x=target_day, color='orange', linestyle='-', alpha=0.8)
    for d in selected_days:
        ax2.axvline(x=d, color='purple', linestyle=':', alpha=0.6)
        # ax2.scatter([d], [correlations[d]['r_squared']], color='red', s=100, zorder=5)
    ax2.set_ylabel('R² (결정계수)')
    ax2.set_title('정보 중복도 (R²)')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(days, igs, alpha=0.3, color='green')
    ax3.plot(days, igs, 'g-', linewidth=1)
    ax3.axvline(x=target_day, color='orange', linestyle='-', alpha=0.8)
    for d in selected_days:
        ax3.axvline(x=d, color='purple', linestyle=':', alpha=0.6)
        # ax3.scatter([d], [correlations[d]['information_gain']], color='red', s=100, zorder=5,
        #            label=f'T={d}' if d == selected_days[0] else None)
    ax3.set_xlabel('거래일 (T)')
    ax3.set_ylabel('Information Gain (1 - R²)')
    ax3.set_title('Information Gain (새로운 정보 비율)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'source_task_correlation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"-> 저장: {path}")


def _plot_selected_pairwise(pivot, target_day, selected_days, output_dir):
    """선택된 Source Tasks 간 pairwise 상관관계 히트맵"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import koreanize_matplotlib
    except ImportError:
        pass

    all_days = [target_day] + selected_days
    labels = [f'T={d}\n(Target)' if d == target_day else f'T={d}' for d in all_days]

    r2_matrix = compute_pairwise_r_squared(pivot, all_days)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax)
    ax.set_title('선택된 Task 간 R² (정보 중복도)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'source_task_pairwise_r2.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"-> 저장: {path}")


def run_select(args):
    """Source Task 선정 (R² 기반 Information Gain)"""
    print("=" * 60)
    print("Source Task 선정 (R² 기반 Information Gain)")
    print("=" * 60)
    print(f"  Target Day: {args.target_day}")

    # 1. BHAR 추세 데이터 로드
    pivot = load_bhar_trend(args.input)
    
    # T=150까지만 사용 (사용자 요청)
    pivot = pivot.loc[:, pivot.columns <= 150]
    print(f"Filtered BHAR Trend Data: T <= 150 (Total {len(pivot.columns)} days)")
    
    # 2. 상관관계 분석 (Target: T={args.target_day})...")
    print(f"\n상관관계 분석 (Target: T={args.target_day})...")
    correlations = compute_correlations(pivot, args.target_day)
    print(f"  유효한 시점: {len(correlations)}개")

    selected = []

    if args.manual_select:
        print(f"\n[Manual Mode] 사용자 지정(CLI) Source Tasks: {args.manual_select}")
        for day in args.manual_select:
            if day in correlations:
                info = correlations[day]
                selected.append({
                    'day': day,
                    'abs_corr': abs(info['correlation']),
                    **info
                })
            else:
                print(f"  [Warning] T={day} 데이터가 없거나 유효하지 않아 제외됨.")
    else:
        print(f"  상관계수 범위: [{args.corr_min}, {args.corr_max}]")
        print(f"  최대 Source 수: {args.max_sources}")
        print(f"  Source 간 최대 R²: {args.max_r2}")

        print(f"\nSource Task Greedy Selection...")
        selected = greedy_select_sources(
            correlations, pivot, args.target_day,
            corr_min=args.corr_min, corr_max=args.corr_max,
            max_pairwise_r2=args.max_r2,
            max_sources=args.max_sources, verbose=True,
        )

    if not selected:
        print("[ERROR] Source Task를 선정할 수 없습니다. 상관계수 범위를 조정하세요.")
        return

    selected_days = [s['day'] for s in selected]

    print(f"\n{'=' * 60}")
    print(f"선정된 Source Tasks ({len(selected)}개):")
    print(f"{'=' * 60}")
    for s in selected:
        print(f"  T={s['day']:3d}: corr={s['correlation']:.3f}, R²={s['r_squared']:.3f}")

    # 결과 저장
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'target_day': args.target_day,
        'source_days': selected_days,
        'correlations': {str(s['day']): round(s['correlation'], 4) for s in selected},
        'r_squared': {str(s['day']): round(s['r_squared'], 4) for s in selected},
        'information_gain': {str(s['day']): round(s['information_gain'], 4) for s in selected},
        'selection_params': {
            'corr_min': args.corr_min,
            'corr_max': args.corr_max,
            'max_pairwise_r2': args.max_r2,
            'max_sources': args.max_sources
        }
    }

    output_path = os.path.join(output_dir, 'source_tasks.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n-> 저장: {output_path}")

    # 시각화
    figures_dir = os.path.join(output_dir, 'figures', 'source_task')
    _plot_correlation_heatmap(correlations, args.target_day, selected_days, figures_dir,
                              corr_min=args.corr_min, corr_max=args.corr_max)
    _plot_selected_pairwise(pivot, args.target_day, selected_days, figures_dir)

    print(f"\n완료!")


# ============================================================
# main
# ============================================================

def main(mode=None):
    parser = argparse.ArgumentParser(
        description='Source Task 선정 (최적화 + 선정 통합)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/source_tasks.py --mode auto
  python scripts/source_tasks.py --mode optimize --trials 50
  python scripts/source_tasks.py --mode select --manual-select 1 3 68
""")
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['optimize', 'select', 'auto'],
                        help='실행 모드 (default: auto)')
    parser.add_argument('--trials', '-n', type=int, default=30,
                        help='Optuna 탐색 횟수 (optimize 모드)')

    # select 모드 옵션
    parser.add_argument('--input', '-i', type=str, default='output/ipo_bhar_trend_160d.csv',
                        help='BHAR 추세 CSV 경로')
    parser.add_argument('--target-day', '-t', type=int, default=PREDICTION_HORIZON,
                        help=f'Target Task 거래일 (기본값: {PREDICTION_HORIZON})')
    parser.add_argument('--max-sources', type=int, default=3,
                        help='최대 Source Task 수')
    parser.add_argument('--corr-min', type=float, default=0.2,
                        help='상관계수 하한')
    parser.add_argument('--corr-max', type=float, default=0.7,
                        help='상관계수 상한')
    parser.add_argument('--max-r2', type=float, default=0.5,
                        help='Source 간 최대 R²')
    parser.add_argument('--manual-select', type=int, nargs='+',
                        help='수동으로 Source Task 날짜 지정')

    # 4_source_optimize.py 결과 자동 적용 (select 모드 단독 실행 시에만)
    params_path = 'output/best_source_params.json'
    # mode를 먼저 확인 (auto 모드에서는 optimize가 파일을 생성하므로 로드 불필요)
    pre_args, _ = parser.parse_known_args()

    # 키워드 인자가 있으면 mode 덮어쓰기
    effective_mode = mode if mode is not None else pre_args.mode

    if effective_mode == 'select' and os.path.exists(params_path):
        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            parser.set_defaults(**best_params)
            print(f"\n[Info] 최적 파라미터 자동 로드 ('{params_path}'):")
            for k, v in best_params.items():
                print(f"  - {k}: {v}")
        except Exception as e:
            print(f"\n[Warning] 파라미터 자동 로드 실패: {e}")

    args = parser.parse_args()
    # 키워드 인자가 있으면 mode 덮어쓰기
    if mode is not None:
        args.mode = mode

    if args.mode == 'optimize':
        run_optimize(args.trials)
    elif args.mode == 'select':
        run_select(args)
    elif args.mode == 'auto':
        print("=" * 60)
        print("  [Auto] Step 1/2: Source Task 파라미터 최적화")
        print("=" * 60)
        run_optimize(args.trials)

        print("\n\n")
        print("=" * 60)
        print("  [Auto] Step 2/2: Source Task 선정")
        print("=" * 60)
        # 최적화 결과 재로드
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                best = json.load(f)
            args.corr_min = best.get('corr_min', args.corr_min)
            args.corr_max = best.get('corr_max', args.corr_max)
            args.max_r2 = best.get('max_pairwise_r2', args.max_r2)
            args.max_sources = best.get('max_sources', args.max_sources)
        run_select(args)


if __name__ == "__main__":
    main()
