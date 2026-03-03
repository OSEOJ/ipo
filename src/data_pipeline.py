"""
Unified Data Pipeline for IPO Prediction System

모든 학습 스크립트(4, 7, 8, 9)와 model_trainer가 공유하는
데이터 로딩/분할/인코딩/스케일링 로직을 통합한 클래스.

사용 예시:
    pipeline = IPODataPipeline()
    pipeline.load()
    X_train_raw, X_test_raw, y_train_combined, y_test_combined = pipeline.get_train_test()
    X_train, X_test = pipeline.process(X_train_raw, X_test_raw)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import MODEL_TEST_SIZE, SCALER
from .utils import encode_post_split, scale_features


class IPODataPipeline:
    """
    IPO 예측 시스템의 데이터 파이프라인.
    
    흐름:
        load() → get_train_test() → [get_cv_folds() or 직접 사용]
                                          ↓
                                   process(X_tr, X_val, shap_features, scale)
                                          ↓
                                   prepare_labels(y_combined, y_source_cols)
    """

    def __init__(self, data_path='output/preprocess/preprocessed_data.csv',
                 test_size=MODEL_TEST_SIZE, n_folds=3):
        self.data_path = data_path
        self.test_size = test_size
        self.n_folds = n_folds

        # load() 호출 후 저장
        self._df = None
        self._X = None
        self._y = None
        self._y_source = None
        self._listing_dates = None
        self._stock_codes = None

    # ================================================================
    # 1. load() - CSV 로드, X/Y/Y_source/상장일/종목코드 분리
    # ================================================================
    def load(self):
        """
        전처리된 데이터 로드.
        
        Returns:
            X (DataFrame): 피처 (종목코드, 상장일, Y, Y_T* 제외)
            y (Series): 타겟 라벨
            y_source (dict): {col_name: Series} 형태의 Source Task 라벨
            listing_dates (Series): 상장일
            stock_codes (Series or None): 종목코드 (있을 경우)
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"전처리된 데이터가 없습니다: {self.data_path}\n"
                f"먼저 'python main.py source_task'을 실행하세요."
            )

        print(f"[Pipeline] Loading: {self.data_path}")
        df = pd.read_csv(self.data_path, encoding='utf-8-sig')

        # Y 분리
        y = df['Y']

        # Source Task Y 분리
        source_cols = [c for c in df.columns if c.startswith('Y_T')]
        y_source = {}
        for col in source_cols:
            y_source[col] = df[col].astype(int)

        # 상장일 분리
        if '상장일' not in df.columns:
            raise ValueError("상장일 컬럼이 없습니다. 6_preprocess.py를 먼저 실행하세요.")
        listing_dates = df['상장일']

        # 종목코드 분리 (있을 경우)
        stock_codes = None
        if '종목코드' in df.columns:
            stock_codes = df['종목코드'].astype(str).str.zfill(6)

        # X 분리 (Y, Y_T*, 상장일, 종목코드 제외)
        drop_cols = ['Y', '상장일'] + source_cols
        if '종목코드' in df.columns:
            drop_cols.append('종목코드')
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # 내부 저장
        self._df = df
        self._X = X
        self._y = y
        self._y_source = y_source
        self._listing_dates = listing_dates
        self._stock_codes = stock_codes

        print(f"[Pipeline] Data shape: X={X.shape}, y={y.shape}")
        print(f"[Pipeline] Class distribution: rise={int((y==1).sum())}, fall={int((y==0).sum())}")
        if source_cols:
            print(f"[Pipeline] Source tasks: {source_cols}")

        return X, y, y_source, listing_dates, stock_codes

    # ================================================================
    # 2. get_train_test() - 상장일 기준 시계열 80/20 분할
    # ================================================================
    def get_train_test(self, X=None, y=None, y_source=None, listing_dates=None):
        """
        상장일 기준 시계열 Train/Test 분할.
        sort_values 방식으로 통일하여 NaT 안전하게 처리.
        
        Args:
            X, y, y_source, listing_dates: 직접 전달 가능. None이면 load()에서 저장된 값 사용.
            
        Returns:
            X_train_raw (DataFrame): 인코딩/스케일링 전 Train 피처
            X_test_raw (DataFrame): 인코딩/스케일링 전 Test 피처
            y_train_combined (DataFrame): Train의 Y + Y_T* 결합
            y_test_combined (DataFrame): Test의 Y + Y_T* 결합
        """
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        y_source = y_source if y_source is not None else self._y_source
        listing_dates = listing_dates if listing_dates is not None else self._listing_dates

        if X is None or y is None:
            raise RuntimeError("데이터가 로드되지 않았습니다. load()를 먼저 호출하세요.")

        # y_combined 구성
        all_y_cols = {'Y': y}
        if y_source:
            all_y_cols.update(y_source)
        y_combined = pd.DataFrame(all_y_cols)

        # 상장일 기준 정렬 (sort_values 방식 통일)
        dates = pd.to_datetime(listing_dates, errors='coerce')
        
        # NaT 체크
        nat_count = dates.isna().sum()
        if nat_count > 0:
            print(f"[Pipeline] Warning: {nat_count}개 상장일 NaT 발견, 해당 행 제거")
            valid_mask = dates.notna()
            X = X.loc[valid_mask].copy()
            y_combined = y_combined.loc[valid_mask].copy()
            dates = dates.loc[valid_mask].copy()

        # 상장일 기준 정렬
        sort_idx = dates.argsort()
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_combined_sorted = y_combined.iloc[sort_idx].reset_index(drop=True)
        dates_sorted = dates.iloc[sort_idx].reset_index(drop=True)

        # 분할
        n_total = len(X_sorted)
        n_test = int(n_total * self.test_size)
        n_train = n_total - n_test

        X_train_raw = X_sorted.iloc[:n_train].reset_index(drop=True)
        X_test_raw = X_sorted.iloc[n_train:].reset_index(drop=True)
        y_train_combined = y_combined_sorted.iloc[:n_train].reset_index(drop=True)
        y_test_combined = y_combined_sorted.iloc[n_train:].reset_index(drop=True)

        train_dates = dates_sorted.iloc[:n_train]
        test_dates = dates_sorted.iloc[n_train:]
        print(f"[Pipeline] Time-based Split:")
        print(f"  Train: {train_dates.min().date()} ~ {train_dates.max().date()} ({n_train}건)")
        print(f"  Test:  {test_dates.min().date()} ~ {test_dates.max().date()} ({n_test}건)")

        return X_train_raw, X_test_raw, y_train_combined, y_test_combined

    # ================================================================
    # 3. get_cv_folds() - Train 내부 시계열 CV
    # ================================================================
    def get_cv_folds(self, X_train_raw, y_train_combined, n_folds=None):
        """
        Train 데이터 내부 TimeSeriesSplit CV.
        데이터를 (n_folds + 3) 등분하여, 최소 학습 데이터가 전체의 50%가 되도록 보장.
        
        예시 (n_folds=3, 700건):
          6등분 (~117건씩)
          Fold 1: [1+2+3] Train 350 → [4] Val 117
          Fold 2: [1+2+3+4] Train 467 → [5] Val 117
          Fold 3: [1+2+3+4+5] Train 583 → [6] Val 117
        
        Args:
            X_train_raw: Train 피처 (인코딩/스케일링 전)
            y_train_combined: Train Y+Source 결합 DataFrame
            n_folds: fold 수 (None이면 self.n_folds 사용)
            
        Yields:
            (fold_idx, X_tr, X_val, y_tr_combined, y_val_combined)
        """
        n_folds = n_folds or self.n_folds
        n_blocks = n_folds + 3  # 3-fold → 6등분, 5-fold → 8등분
        val_size = len(X_train_raw) // n_blocks
        tscv = TimeSeriesSplit(n_splits=n_folds, test_size=val_size)

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train_raw)):
            X_tr = X_train_raw.iloc[tr_idx].reset_index(drop=True)
            X_val = X_train_raw.iloc[val_idx].reset_index(drop=True)
            y_tr = y_train_combined.iloc[tr_idx].reset_index(drop=True)
            y_val = y_train_combined.iloc[val_idx].reset_index(drop=True)

            print(f"  [Fold {fold_idx+1}/{n_folds}] "
                  f"Train: {len(X_tr)}, Val: {len(X_val)}")

            yield fold_idx, X_tr, X_val, y_tr, y_val

    # ================================================================
    # 4. process() - 인코딩 + (SHAP필터) + 스케일링
    # ================================================================
    def process(self, X_train, X_test, shap_features=None, scale=True, verbose=True):
        """
        인코딩 → (선택적 SHAP 필터) → (선택적 스케일링) 파이프라인.
        
        주의: SHAP 피처 목록은 인코딩 후 피처 이름 기준이어야 합니다.
              (예: '업종'이 아닌 '업종_기타', '업종_제조업' 등)
        
        Args:
            X_train: Train 피처 (raw, 인코딩 전)
            X_test: Test 피처 (raw, 인코딩 전)
            shap_features: SHAP 선택 피처 목록 (None이면 필터 미적용)
            scale: True이면 스케일링 적용
            verbose: 인코딩 로그 출력 여부
            
        Returns:
            X_train_processed, X_test_processed (DataFrame)
        """
        # 1. 인코딩 (업종 그룹핑, ROA/ROE 처리, 결측치 대체)
        X_train_enc, X_test_enc = encode_post_split(X_train, X_test, verbose=verbose)

        # 2. SHAP 피처 필터링
        if shap_features:
            current_cols = X_train_enc.columns.tolist()
            valid_features = [c for c in shap_features if c in current_cols]
            if valid_features:
                X_train_enc = X_train_enc[valid_features]
                X_test_enc = X_test_enc[valid_features]
                if verbose:
                    print(f"  [Pipeline] SHAP Filter: {len(current_cols)} → {len(valid_features)} columns")
            else:
                if verbose:
                    print("  [Pipeline] Warning: SHAP features provided but no match found.")

        # 3. 스케일링
        if scale:
            X_train_enc, X_test_enc = scale_features(X_train_enc, X_test_enc, scaler_type=SCALER)

        return X_train_enc, X_test_enc

    # ================================================================
    # 5. prepare_labels() - y_dict 구성 (MTL용)
    # ================================================================
    @staticmethod
    def prepare_labels(y_combined, y_source_cols=None):
        """
        y_combined DataFrame에서 target/source 라벨 딕셔너리 구성.
        
        Args:
            y_combined: DataFrame with 'Y' + 'Y_T*' columns
            y_source_cols: source 컬럼 리스트. None이면 'Y_T'로 시작하는 모든 컬럼.
            
        Returns:
            y_dict: {'target': y, 'source_0': y_s0, ...}
            y_target: target Series
        """
        y_target = y_combined['Y']

        if y_source_cols is None:
            y_source_cols = [c for c in y_combined.columns if c.startswith('Y_T')]

        y_dict = {'target': y_target}
        for i, col in enumerate(y_source_cols):
            if col in y_combined.columns:
                y_dict[f'source_{i}'] = y_combined[col]

        return y_dict, y_target

    # ================================================================
    # 6. load_shap_features() - SHAP 피처 목록 로드
    # ================================================================
    @staticmethod
    def load_shap_features(shap_path='output/shap_selected_features.json'):
        """
        SHAP 선정 피처 목록 로드.
        
        Returns:
            list of feature names, or None if file not found
        """
        if os.path.exists(shap_path):
            with open(shap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            features = data.get('selected_features', [])
            print(f"[Pipeline] SHAP features loaded: {len(features)} features")
            return features
        return None

    # ================================================================
    # Helper: Train 내부 단일 시계열 분할 (RFE용)
    # ================================================================
    @staticmethod
    def split_train_internal(X_train, y_train_combined, val_ratio=0.2):
        """
        Train 데이터 내부에서 시계열 기반 단일 Train/Val 분할.
        이미 시간순 정렬된 상태를 전제.
        
        Args:
            X_train: Train 피처
            y_train_combined: Train Y DataFrame
            val_ratio: Validation 비율 (기본 0.2)
            
        Returns:
            X_tr, X_val, y_tr_combined, y_val_combined
        """
        n = len(X_train)
        n_val = int(n * val_ratio)
        n_tr = n - n_val

        X_tr = X_train.iloc[:n_tr].reset_index(drop=True)
        X_val = X_train.iloc[n_tr:].reset_index(drop=True)
        y_tr = y_train_combined.iloc[:n_tr].reset_index(drop=True)
        y_val = y_train_combined.iloc[n_tr:].reset_index(drop=True)

        print(f"  [Pipeline] Internal Split: Train {n_tr} / Val {n_val}")
        return X_tr, X_val, y_tr, y_val

    # ================================================================
    # Helper: Train 분할 시 stock_codes도 함께 분할
    # ================================================================
    def get_train_test_with_codes(self):
        """
        get_train_test()와 동일하지만 stock_codes도 Train/Test로 분할하여 반환.
        4_source_optimize.py처럼 종목코드가 필요한 경우 사용.
        
        Returns:
            X_train_raw, X_test_raw, y_train_combined, y_test_combined,
            codes_train, codes_test
        """
        if self._stock_codes is None:
            raise ValueError("종목코드가 없습니다. 전처리 데이터를 확인하세요.")

        listing_dates = self._listing_dates
        dates = pd.to_datetime(listing_dates, errors='coerce')

        # NaT 필터
        valid_mask = dates.notna()
        X = self._X.loc[valid_mask].copy()
        y_source = {k: v.loc[valid_mask] for k, v in self._y_source.items()}
        y = self._y.loc[valid_mask].copy()
        dates = dates.loc[valid_mask].copy()
        codes = self._stock_codes.loc[valid_mask].copy()

        # y_combined
        all_y_cols = {'Y': y}
        all_y_cols.update(y_source)
        y_combined = pd.DataFrame(all_y_cols)

        # 정렬
        sort_idx = dates.argsort()
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_combined_sorted = y_combined.iloc[sort_idx].reset_index(drop=True)
        codes_sorted = codes.iloc[sort_idx].reset_index(drop=True)

        # 분할
        n_total = len(X_sorted)
        n_test = int(n_total * self.test_size)
        n_train = n_total - n_test

        X_train = X_sorted.iloc[:n_train].reset_index(drop=True)
        X_test = X_sorted.iloc[n_train:].reset_index(drop=True)
        y_train = y_combined_sorted.iloc[:n_train].reset_index(drop=True)
        y_test = y_combined_sorted.iloc[n_train:].reset_index(drop=True)
        codes_train = codes_sorted.iloc[:n_train].reset_index(drop=True)
        codes_test = codes_sorted.iloc[n_train:].reset_index(drop=True)

        return X_train, X_test, y_train, y_test, codes_train, codes_test
