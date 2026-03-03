"""
Optuna 하이퍼파라미터 탐색 (AITM 모델)

벤치마크와 동일한 80/20 시계열 분할 후,
Train 80% 내부에서 3-Fold TimeSeriesSplit CV로 하이퍼파라미터를 탐색합니다.
Test 20%는 Optuna에서 사용하지 않아 벤치마크 결과의 신뢰성을 보장합니다.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.metrics import f1_score


# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_RANDOM_STATE, MTL_MANUAL_SOURCE_TASKS, MLP_PARAMS, OPTUNA_SEARCH_SPACE,
)
from src.utils import set_seed
from src.models.aitm import AITMClassifier
from src.data_pipeline import IPODataPipeline


N_CV_FOLDS = 3  # 3-Fold TimeSeriesSplit CV


def load_data():
    """
    전처리된 데이터를 로드하고, 벤치마크와 동일한 시계열 80/20 분할 수행.
    Train 80%만 반환하여 Optuna가 Test 20%를 건드리지 않도록 합니다.
    """
    pipeline = IPODataPipeline()
    X, y, y_source, listing_dates, _stock_codes = pipeline.load()

    # SHAP 피처 로드
    shap_features = pipeline.load_shap_features()

    # Train/Test 분할 (Test는 사용하지 않음)
    X_train_raw, _X_test, y_train_combined, _y_test = pipeline.get_train_test()

    print(f"Optuna Data (Train only): X={X_train_raw.shape}")

    return pipeline, X_train_raw, y_train_combined, y_source, shap_features


def objective(trial, pipeline, X_train_all, y_train_combined, y_source, shap_features):
    """Optuna Objective: 3-Fold TimeSeriesSplit CV로 F1 Score 최대화"""

    # --- 탐색 공간 정의 (from config.py) ---
    sp = OPTUNA_SEARCH_SPACE
    
    n_layers = trial.suggest_int('n_layers', sp['n_layers']['low'], sp['n_layers']['high'])
    if n_layers == 1:
        hidden_dims = [trial.suggest_categorical('h1', sp['h1'])]
    else:
        h1 = trial.suggest_categorical('h1', sp['h1'])
        h2 = trial.suggest_categorical('h2', sp['h2'])
        hidden_dims = [h1, h2]

    tower_dim = trial.suggest_categorical('tower_dim', sp['tower_dim'])
    
    dropout = trial.suggest_float('dropout', sp['dropout']['low'], sp['dropout']['high'], step=sp['dropout']['step'])
    learning_rate = trial.suggest_float('learning_rate', sp['learning_rate']['low'], sp['learning_rate']['high'], log=sp['learning_rate']['log'])
    weight_decay = trial.suggest_float('weight_decay', sp['weight_decay']['low'], sp['weight_decay']['high'], log=sp['weight_decay']['log'])
    batch_size = trial.suggest_categorical('batch_size', sp['batch_size'])
    
    focal_gamma = trial.suggest_float('focal_gamma', sp['focal_gamma']['low'], sp['focal_gamma']['high'], step=sp['focal_gamma']['step'])
    source_loss_weight = trial.suggest_float('source_loss_weight', sp['source_loss_weight']['low'], sp['source_loss_weight']['high'], step=sp['source_loss_weight']['step'])
    
    epochs = 300  # config 700에서 벗어나 Optuna 전용으로 고정

    # --- 3-Fold TimeSeriesSplit CV ---
    fold_scores = []
    source_cols = [c for c in y_train_combined.columns if c.startswith('Y_T')]

    for fold_idx, X_tr_raw, X_val_raw, y_tr_combined, y_val_combined in pipeline.get_cv_folds(
        X_train_all, y_train_combined, n_folds=N_CV_FOLDS
    ):
        # 인코딩 + SHAP 필터 + 스케일링 (fold별)
        X_tr, X_val = pipeline.process(X_tr_raw, X_val_raw, shap_features=shap_features, verbose=False)

        # y 분리
        y_tr = y_tr_combined['Y'].values
        y_val = y_val_combined['Y'].values



        # y_train_dict 구성 (target + source) — CV fold 내부 데이터이므로 직접 구성
        y_tr_dict = {'target': y_tr}
        for i, col in enumerate(source_cols):
            if col in y_tr_combined.columns:
                y_tr_dict[f'source_{i}'] = y_tr_combined[col].values

        # 모델 생성
        model = AITMClassifier(
            bottom_mlp_dims=hidden_dims,
            tower_mlp_dims=[tower_dim],
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            source_days=list(MTL_MANUAL_SOURCE_TASKS),
            verbose=False,
            weight_decay=weight_decay,
            focal_gamma=focal_gamma,
            source_loss_weight=source_loss_weight,
            early_stopping_patience=100,
        )

        # 학습 (Epoch 단위 Pruning 보고 제거 -> Step 중복 경고 방지)
        model.fit(X_tr, y_tr_dict,
                  X_valid=X_val, y_valid_dict={'target': y_val})

        # 평가: Threshold Moving으로 Best F1
        proba = model.predict_proba(X_val)[:, 1]

        best_f1 = 0
        for thr in np.arange(0.05, 0.95, 0.01):
            preds = (proba >= thr).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        fold_scores.append(best_f1)

        # Pruning (fold 단위)
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_f1 = np.mean(fold_scores)
    return mean_f1


def main(n_trials=None):
    # 키워드 인자가 없으면 argparse로 CLI 파싱
    if n_trials is None:
        parser = argparse.ArgumentParser(description='Optuna HP Search')
        parser.add_argument('--n_trials', type=int, default=100, help='탐색 횟수')
        args = parser.parse_args()
        n_trials = args.n_trials

    print("=" * 60)
    print(f"Optuna HP Search (AITM, {n_trials} trials, {N_CV_FOLDS}-Fold CV)")
    print("=" * 60)

    set_seed(MODEL_RANDOM_STATE)

    pipeline, X_train_all, y_train_combined, y_source, shap_features = load_data()

    # Optuna Study 생성
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        study_name='aitm_hp_search',
    )

    # 현재 설정을 첫 번째 Trial로 등록 (baseline) - 새로운 공간에 맞게 꼼꼼하게 다시 조정
    study.enqueue_trial({
        'n_layers': 2, 'h1': 64, 'h2': 64, 'tower_dim': 32,
        'dropout': 0.3, 'learning_rate': 1e-3, 'weight_decay': 1e-6,
        'batch_size': 64, 'focal_gamma': 2.0,
        'source_loss_weight': 0.4,
    })

    study.optimize(
        lambda trial: objective(trial, pipeline, X_train_all, y_train_combined, y_source, shap_features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # 결과 출력
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best F1 (CV Mean): {best.value:.4f}")
    print("=" * 60)
    print("Best Hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # 결과 저장
    output_path = Path('output/optuna_best_params.json')
    result = {
        'best_f1': round(best.value, 4),
        'best_params': best.params,
        'n_trials': n_trials,
        'n_cv_folds': N_CV_FOLDS,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n-> Saved to {output_path}")

    # Top 5 출력
    print(f"\nTop 5 Trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        print(f"  #{t.number}: F1={t.value:.4f} | {t.params}")


if __name__ == '__main__':
    main()
