"""
유틸리티 함수 모듈
"""
import random
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def parse_date(date_str: Optional[str], fmt: str = '%Y-%m-%d') -> Optional[datetime]:
    """
    날짜 문자열을 datetime 객체로 변환

    Args:
        date_str: 날짜 문자열
        fmt: 날짜 형식 (기본값: '%Y-%m-%d')

    Returns:
        datetime 객체 또는 None (실패 시)
    """
    if not date_str:
        return None

    try:
        return datetime.strptime(date_str, fmt)
    except (ValueError, TypeError):
        return None


def set_seed(seed=42):
    """
    모든 라이브러리의 Random Seed 고정
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random Seed set to {seed}")


def _get_scaler(scaler_type):
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    else:
        return RobustScaler()


# 스케일링 대상 컬럼 정의
LOG_SCALE_FEATURES = [
    '총자산(요약재무)', '부채(요약재무)', '자본금(요약재무)',
    '매출액(영업수익)', '영업이익', '당기순이익(포괄손익계산서)',
    '영업활동으로인한현금흐름', '공모가 (원)', '공모금액 (천원)',
    '최초상장주식수 (주)', '기관경쟁률', '개인경쟁률',
]

RATIO_FEATURES = [
    '부채비율(표준재무)', '유동비율(표준재무)', '현금비율(표준재무)',
    '유동부채비율(표준재무)', '자기자본비율(표준재무)',
    'ROA(표준재무)', 'ROE(표준재무)', 'EBITDA마진율(표준재무)',
    '순이익률(표준재무)', '의무보유확약', '시장지수_15일_수익률',
    '총차입금/EBITDA(표준재무)', '이자보상배율(표준재무)',
    '재고자산회전율(표준재무)', '총자산회전율(표준재무)',
    '총자산증가율(표준재무)', '유동자산증가율(표준재무)',
    '당좌자산증가율(표준재무)', '자기자본증가율(표준재무)',
    '기관배정', '업력',
]


def scale_features(X_train, X_test, scaler_type='standard'):
    """
    Train 데이터로 scaler를 fit하고, Train/Test 모두 transform.
    데이터 누수를 방지하기 위해 반드시 split 이후에 호출해야 합니다.

    Args:
        X_train: 학습 데이터 (DataFrame)
        X_test: 테스트 데이터 (DataFrame)
        scaler_type: 스케일러 종류 ('standard', 'robust', 'minmax')

    Returns:
        X_train_scaled, X_test_scaled (DataFrame)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 1단계: 로그 변환 (데이터 의존 없이 동일하게 적용)
    for col in X_train.columns:
        if col in LOG_SCALE_FEATURES:
            if col == '기관경쟁률':
                for df in [X_train, X_test]:
                    mask = df[col] != -1
                    df.loc[mask, col] = np.sign(df.loc[mask, col]) * np.log1p(np.abs(df.loc[mask, col]))
            else:
                for df in [X_train, X_test]:
                    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # 2단계: 스케일링 (train fit → train/test transform)
    for col in X_train.columns:
        if col in LOG_SCALE_FEATURES:
            scaler = _get_scaler(scaler_type)
            if col == '기관경쟁률':
                train_mask = X_train[col] != -1
                test_mask = X_test[col] != -1
                if train_mask.sum() > 0:
                    scaler.fit(X_train.loc[train_mask, [col]])
                    X_train.loc[train_mask, col] = scaler.transform(X_train.loc[train_mask, [col]]).flatten()
                    if test_mask.sum() > 0:
                        X_test.loc[test_mask, col] = scaler.transform(X_test.loc[test_mask, [col]]).flatten()
            else:
                scaler.fit(X_train[[col]])
                X_train[col] = scaler.transform(X_train[[col]]).flatten()
                X_test[col] = scaler.transform(X_test[[col]]).flatten()

        elif col in RATIO_FEATURES:
            scaler = _get_scaler(scaler_type)
            scaler.fit(X_train[[col]])
            X_train[col] = scaler.transform(X_train[[col]]).flatten()
            X_test[col] = scaler.transform(X_test[[col]]).flatten()

        elif col.endswith('_수치'):
            scaler = _get_scaler(scaler_type)
            scaler.fit(X_train[[col]])
            X_train[col] = scaler.transform(X_train[[col]]).flatten()
            X_test[col] = scaler.transform(X_test[[col]]).flatten()

        elif X_train[col].nunique() > 2 and not col.startswith('업종_'):
            if X_train[col].max() - X_train[col].min() > 100:
                scaler = _get_scaler(scaler_type)
                scaler.fit(X_train[[col]])
                X_train[col] = scaler.transform(X_train[[col]]).flatten()
                X_test[col] = scaler.transform(X_test[[col]]).flatten()

    # 기관경쟁률 결측(-1)을 0(평균)으로 변환
    if '기관경쟁률' in X_train.columns:
        X_train['기관경쟁률'] = X_train['기관경쟁률'].replace(-1, 0)
        X_test['기관경쟁률'] = X_test['기관경쟁률'].replace(-1, 0)

    return X_train, X_test


# ============================================================
# Split 이후 인코딩 (데이터 누수 방지)
# ============================================================

def encode_post_split(X_train, X_test, industry_min_count=15, verbose=True):
    """
    Train/Test Split 이후, train 데이터만으로 데이터 의존적 변환을 수행합니다.
    반드시 scale_features() 호출 전에 실행해야 합니다.

    처리 항목:
      1. ROA/ROE 0값 → train 비영 평균으로 대체
      2. 업종 그룹핑 + 원핫인코딩 (train 빈도 기반)
      3. 결측치 → train 중앙값/최빈값으로 대체

    Args:
        X_train: 학습 데이터 (DataFrame)
        X_test: 테스트 데이터 (DataFrame)
        industry_min_count: 주요 업종 판별 빈도 기준 (기본값: 15)

    Returns:
        X_train, X_test (DataFrame) - 인코딩 완료
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # --- 1. ROA/ROE 0값 → train 비영 평균으로 대체 ---
    for col in ['ROA(표준재무)', 'ROE(표준재무)']:
        if col in X_train.columns:
            train_non_zero = X_train[X_train[col] != 0][col]
            if len(train_non_zero) > 0:
                non_zero_mean = train_non_zero.mean()
                zero_count_train = (X_train[col] == 0).sum()
                zero_count_test = (X_test[col] == 0).sum()
                X_train[col] = X_train[col].replace(0, non_zero_mean)
                X_test[col] = X_test[col].replace(0, non_zero_mean)
                if verbose:
                    print(f"  [PostSplit] {col}: train {zero_count_train}, test {zero_count_test} zeros → mean={non_zero_mean:.2f}")

    # --- 2. 업종 그룹핑 + 원핫인코딩 (train 빈도 기반) ---
    if '업종' in X_train.columns:
        # train에서만 주요 업종 결정
        train_counts = X_train['업종'].value_counts()
        major_industries = train_counts[train_counts > industry_min_count].index.tolist()
        if verbose:
            print(f"  [PostSplit] 업종: 주요 {len(major_industries)}개 (train 빈도>{industry_min_count})")

        for df in [X_train, X_test]:
            df['업종_grouped'] = df['업종'].apply(
                lambda x: x if x in major_industries else '기타'
            )

        # train 기준 더미 컬럼 생성
        train_dummies = pd.get_dummies(X_train['업종_grouped'], prefix='업종')
        test_dummies = pd.get_dummies(X_test['업종_grouped'], prefix='업종')

        # test에 없는 컬럼은 0으로, train에 없는 컬럼은 제거
        for col_name in train_dummies.columns:
            if col_name not in test_dummies.columns:
                test_dummies[col_name] = 0
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        # 원본 업종 컬럼 제거, 더미 컬럼 추가
        X_train = X_train.drop(columns=['업종', '업종_grouped'])
        X_test = X_test.drop(columns=['업종', '업종_grouped'])
        X_train = pd.concat([X_train, train_dummies], axis=1)
        X_test = pd.concat([X_test, test_dummies], axis=1)

    # --- 3. 결측치 → train 중앙값/최빈값으로 대체 ---
    fill_values = {}
    for col in X_train.columns:
        if X_train[col].isna().any() or X_test[col].isna().any():
            unique_vals = X_train[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                # 이진 피처 → 최빈값
                mode_vals = X_train[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 0
            else:
                # 수치형 피처 → 중앙값
                fill_val = X_train[col].median()
            fill_values[col] = fill_val

    if fill_values:
        X_train = X_train.fillna(fill_values)
        X_test = X_test.fillna(fill_values)
        if verbose:
            print(f"  [PostSplit] 결측치 대체: {len(fill_values)}개 컬럼 (train 기반 중앙값/최빈값)")

    return X_train, X_test


def evaluate_with_threshold(y_true, y_pred_proba):
    """
    Threshold Moving 기반 모델 평가 (공통 함수).
    roc_curve를 이용하여 Youden's J statistic(TPR - FPR)이 최대가 되는 최적의 threshold를 찾고, 
    주요 지표를 반환합니다.
    
    Args:
        y_true: 실제 라벨 (array-like)
        y_pred_proba: 양성 클래스 확률 (array-like)
        
    Returns:
        dict: {'AUC', 'F1', 'Precision', 'Recall', 'Accuracy', 'Threshold'}
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, roc_curve
    )

    # [변경] 방안 3: Youden's J statistic (J = Sensitivity + Specificity - 1 = TPR - FPR)
    # ROC 커브 상에서 (좌상단 모서리인) 완벽한 분류기에 가장 가까운 점을 찾는 통계적으로 가장 균형 잡힌 방법입니다.
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    # 간혹 threshold가 1 이상으로 나오는 경우를 대비
    best_threshold = thresholds[best_idx]
    if best_threshold > 1.0:
        best_threshold = thresholds[1] if len(thresholds) > 1 else 0.5

    y_pred = (np.asarray(y_pred_proba) >= best_threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'AUC': auc,
        'Threshold': best_threshold,
    }

