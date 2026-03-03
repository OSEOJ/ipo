"""
전처리 스크립트
IPO 데이터 전처리를 수행하여 학습 가능한 형태로 저장합니다.

실행: python 6_preprocess.py
출력: output/preprocess/preprocessed_data.csv
"""
import pandas as pd
import numpy as np
import os
import sys
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (FINANCE_DATA_PATH, MISSING_DATA_PATH,
                        PREDICTION_HORIZON,
                        MTL_SOURCE_TASKS_PATH)


# 증가율 컬럼에서 나타나는 상태 텍스트 목록
GROWTH_STATE_KEYWORDS = ['적자전환', '흑자전환', '적자지속', '흑자지속']
CRAWL_DATA_PATH = os.path.join('output', 'crawl', 'ipo_crawl.csv')
CRAWL_COLUMNS = [
    '종목명', '종목코드', '시장구분', '업종', '상장일',
    '기관경쟁률', '의무보유확약', '기관배정', '개인경쟁률',
    '업력', '시장지수_15일_수익률'
]
CORE_FINANCE_COLUMNS = [
    '총자산(요약재무)',
    '매출액(영업수익)',
    '영업이익',
    '당기순이익(포괄손익계산서)',
]


def normalize_stock_code(series):
    """종목코드를 문자열로 정규화합니다 (숫자형은 6자리 zero-padding)."""
    normalized = series.astype(str).str.strip().str.upper()
    normalized = normalized.str.replace('.0', '', regex=False)
    normalized = normalized.replace({'': np.nan, 'NAN': np.nan, 'NONE': np.nan})

    numeric_mask = normalized.str.fullmatch(r'\d+', na=False)
    normalized.loc[numeric_mask] = normalized.loc[numeric_mask].str.zfill(6)
    return normalized


def merge_crawl_data(df, crawl_path=CRAWL_DATA_PATH):
    """재무 데이터와 ipo_crawl.csv를 종목코드 기준으로 병합합니다."""
    if not os.path.exists(crawl_path):
        print(f"[Info] Crawl 데이터 없음: {crawl_path} (병합 생략)")
        return df

    print(f"Loading Crawl Data: {crawl_path}")
    df_crawl = pd.read_csv(crawl_path, encoding='utf-8-sig')
    if df_crawl.empty:
        print("[Info] Crawl 데이터가 비어 있습니다 (병합 생략)")
        return df

    available_cols = [col for col in CRAWL_COLUMNS if col in df_crawl.columns]
    if '종목코드' not in available_cols:
        print("[Warning] Crawl 데이터에 종목코드가 없어 병합 생략")
        return df

    df_crawl = df_crawl[available_cols].copy()
    df_crawl['종목코드'] = normalize_stock_code(df_crawl['종목코드'])
    df_crawl = df_crawl.dropna(subset=['종목코드'])

    if '상장일' in df_crawl.columns:
        crawl_dates = pd.to_datetime(df_crawl['상장일'], errors='coerce')
        df_crawl['_sort_상장일'] = crawl_dates
        df_crawl['상장일'] = crawl_dates.dt.strftime('%Y-%m-%d')
    else:
        df_crawl['_sort_상장일'] = pd.NaT

    for num_col in ['기관경쟁률', '의무보유확약', '기관배정', '개인경쟁률', '업력', '시장지수_15일_수익률']:
        if num_col in df_crawl.columns:
            df_crawl[num_col] = pd.to_numeric(df_crawl[num_col], errors='coerce')

    df_crawl = df_crawl.sort_values('_sort_상장일').drop_duplicates(subset=['종목코드'], keep='last')
    df_crawl = df_crawl.drop(columns=['_sort_상장일'])

    finance_codes = set(df['종목코드'].dropna().astype(str).tolist())
    crawl_codes = set(df_crawl['종목코드'].dropna().astype(str).tolist())
    overlap_count = len(finance_codes & crawl_codes)
    new_count = len(crawl_codes - finance_codes)
    print(f"  Crawl rows: {len(df_crawl)} (기존 매칭 {overlap_count}건, 신규 {new_count}건)")

    before_rows = len(df)
    merged = pd.merge(df, df_crawl, on='종목코드', how='outer', suffixes=('', '_crawl'))

    for col in available_cols:
        if col == '종목코드':
            continue
        crawl_col = f'{col}_crawl'
        if crawl_col not in merged.columns:
            continue

        if col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[crawl_col])
        else:
            merged[col] = merged[crawl_col]

    crawl_suffix_cols = [f'{c}_crawl' for c in available_cols if c != '종목코드']
    merged = merged.drop(columns=[c for c in crawl_suffix_cols if c in merged.columns])

    if '종목명' in merged.columns and '회사명' in merged.columns:
        merged['종목명'] = merged['종목명'].where(merged['종목명'].notna(), merged['회사명'])

    print(f"  Finance + Crawl 병합: {before_rows} → {len(merged)} rows")
    return merged


def drop_rows_without_core_finance(df):
    """핵심 재무 컬럼이 모두 결측인 행을 제거합니다."""
    existing_cols = [col for col in CORE_FINANCE_COLUMNS if col in df.columns]
    if not existing_cols:
        print("[Info] 핵심 재무 컬럼이 없어 결측 행 제거를 건너뜁니다.")
        return df

    all_missing_mask = df[existing_cols].isna().all(axis=1)
    remove_count = int(all_missing_mask.sum())
    if remove_count > 0:
        before = len(df)
        df = df.loc[~all_missing_mask].copy()
        print(f"Removed rows without core finance: {remove_count} ({before} → {len(df)})")
    else:
        print("Removed rows without core finance: 0")

    return df


def split_mixed_column(df, col_name):
    """
    숫자와 텍스트(적자전환, 흑자전환 등)가 혼합된 컬럼을 두 개로 분리합니다.
    """
    if col_name not in df.columns:
        return df, []
    
    new_cols = []
    
    # 1. 수치형 피처 생성
    numeric_col = f"{col_name}_수치"
    df[numeric_col] = pd.to_numeric(
        df[col_name].astype(str).str.replace(',', '').str.replace('%', '').str.strip(),
        errors='coerce'
    ).fillna(0)
    new_cols.append(numeric_col)
    
    # 2. 상태 피처 생성
    def extract_state(val):
        val_str = str(val).strip()
        for keyword in GROWTH_STATE_KEYWORDS:
            if keyword in val_str:
                return keyword
        return '정상'
    
    state_col = f"{col_name}_상태"
    df[state_col] = df[col_name].apply(extract_state)
    
    # 3. One-hot encoding for 상태
    state_dummies = pd.get_dummies(df[state_col], prefix=col_name)
    df = pd.concat([df, state_dummies], axis=1)
    new_cols.extend(state_dummies.columns.tolist())
    
    return df, new_cols


def load_finance_data(finance_path):
    """재무 데이터 로드"""
    print(f"Loading Financial Data: {finance_path}")
    df = pd.read_csv(finance_path, encoding='utf-8-sig')

    # 누락 재무 데이터 추가
    missing_path = str(MISSING_DATA_PATH)
    if os.path.exists(missing_path):
        df_missing = pd.read_excel(missing_path)
        print(f"Loading Missing Data: {missing_path} ({len(df_missing)} rows)")
        df = pd.concat([df, df_missing], ignore_index=True)
        df = df.drop_duplicates(subset=['종목코드'], keep='first')
        print(f"Combined Financial Data: {len(df)} rows")
    
    # 종목코드 통일
    df['종목코드'] = normalize_stock_code(df['종목코드'])
    
    # 종목명 정리
    if '종목명' not in df.columns and '회사명' in df.columns:
        df['종목명'] = df['회사명']

    # IPO Crawl 데이터 병합
    df = merge_crawl_data(df)

    # 핵심 재무 결측 행 제거 (crawl-only 노이즈 방지)
    df = drop_rows_without_core_finance(df)
    
    # Unnamed: 17 삭제
    if 'Unnamed: 17' in df.columns:
        df = df.drop(columns=['Unnamed: 17'])
        print("Dropped 'Unnamed: 17' column")
    
    # === 기관경쟁률 결측 처리 ===
    if '기관경쟁률' in df.columns:
        df['기관경쟁률_존재'] = df['기관경쟁률'].notna().astype(int)
        df['기관경쟁률'] = df['기관경쟁률'].fillna(-1)
        missing_count = (df['기관경쟁률'] == -1).sum()
        print(f"기관경쟁률 missing: {missing_count} rows → filled with -1")
    
    # ROA/ROE 0값 처리는 split 이후 train 기반으로 수행 (encode_post_split)
    for col in ['ROA(표준재무)', 'ROE(표준재무)']:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"{col}: {zero_count} zeros → split 이후 train 기반 대체 예정")
    
    # 필터링
    print(f"Before filtering: {len(df)} rows")
    
    # 스팩 제외
    df = df[~df['종목명'].str.contains('스팩', na=False)]
    print(f"After removing SPACs: {len(df)} rows")
    
    # 시장 필터링
    if '시장구분' in df.columns:
        valid_markets = ['KOSPI', 'KOSDAQ']
        known_mask = df['시장구분'].isin(valid_markets)
        invalid_mask = df['시장구분'].notna() & ~known_mask
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            df = df[~invalid_mask]
            print(f"Removed invalid 시장구분 rows: {invalid_count}")
        df['Market_Encoding'] = (df['시장구분'] == 'KOSPI').astype(int)
    else:
        df['Market_Encoding'] = 0
    
    print(f"After Market filtering: {len(df)} rows")
    
    return df


def _load_source_task_days():
    """source_tasks.json 또는 Config에서 Source Task 날짜 로드"""
    try:
        from src.config import MTL_MANUAL_SOURCE_TASKS
        if MTL_MANUAL_SOURCE_TASKS:
            print(f"  [Config] MTL Source Tasks (Manual Override): {MTL_MANUAL_SOURCE_TASKS}")
            return MTL_MANUAL_SOURCE_TASKS
    except ImportError:
        pass

    source_days = []
    if os.path.exists(str(MTL_SOURCE_TASKS_PATH)):
        try:
            with open(str(MTL_SOURCE_TASKS_PATH), 'r', encoding='utf-8') as f:
                data = json.load(f)
            source_days = data.get('source_days', [])
            print(f"  [File] MTL Source Tasks 로드: {source_days}")
        except Exception as e:
            print(f"[Warning] Source tasks 로드 실패: {e}")
    return source_days


def load_bhar_from_trend(df):
    """
    3_bhar_trend.py가 생성한 output/ipo_bhar_trend_160d.csv에서
    Target(T=PREDICTION_HORIZON) 및 Source Task BHAR을 읽어 병합합니다.
    API 재호출 없이 일관된 BHAR 값을 사용합니다.
    """
    bhar_trend_path = os.path.join('output', 'ipo_bhar_trend_160d.csv')
    if not os.path.exists(bhar_trend_path):
        raise FileNotFoundError(
            f"BHAR trend 파일이 없습니다: {bhar_trend_path}\n"
            f"먼저 'python main.py bhar'을 실행하세요."
        )

    source_days = _load_source_task_days()
    target_days = [PREDICTION_HORIZON] + [d for d in source_days if d != PREDICTION_HORIZON]

    print(f"\nLoading BHAR from {bhar_trend_path}")
    print(f"  Target T={PREDICTION_HORIZON}, Source days={source_days}")

    df_trend = pd.read_csv(bhar_trend_path, encoding='utf-8-sig')
    df_trend['종목코드'] = df_trend['종목코드'].astype(str).str.zfill(6)

    # T=1 시가도 추출 (상장일 시초가 보완용)
    df_t1 = df_trend[df_trend['T'] == 1][['종목코드', '시가']].copy()
    df_t1 = df_t1.rename(columns={'시가': '시가_T1'})

    # 필요한 T값만 필터링하여 pivot
    df_needed = df_trend[df_trend['T'].isin(target_days)].copy()

    # 종목코드 × T → BHAR pivot (wide format)
    pivot = df_needed.pivot_table(
        index='종목코드', columns='T', values='BHAR', aggfunc='first'
    )

    # 컬럼명 변경: Target은 'BHAR', Source는 'BHAR_T{n}'
    rename_map = {}
    for t in pivot.columns:
        if t == PREDICTION_HORIZON:
            rename_map[t] = 'BHAR'
        else:
            rename_map[t] = f'BHAR_T{t}'
    pivot = pivot.rename(columns=rename_map).reset_index()

    # 병합
    before_count = len(df)
    df = pd.merge(df, pivot, on='종목코드', how='inner')
    # 시가_T1 병합 (상장일 시초가 보완용)
    df = pd.merge(df, df_t1, on='종목코드', how='left')
    print(f"  Merged: {before_count} → {len(df)} rows (BHAR 매칭)")

    # 통계 출력
    valid_bhar = df['BHAR'].dropna()
    print(f"\nBHAR Statistics (Target T={PREDICTION_HORIZON}):")
    print(f"  Valid: {len(valid_bhar)}/{len(df)}")
    if len(valid_bhar) > 0:
        print(f"  Mean: {valid_bhar.mean():.4f}")
        print(f"  Std: {valid_bhar.std():.4f}")
        print(f"  Positive: {(valid_bhar >= 0).sum()} ({(valid_bhar >= 0).mean()*100:.1f}%)")

    for sd in source_days:
        if sd == PREDICTION_HORIZON:
            continue
        col = f'BHAR_T{sd}'
        if col in df.columns:
            valid_sd = df[col].dropna()
            print(f"  Source T={sd}: {len(valid_sd)}/{len(df)} valid, mean={valid_sd.mean():.4f}")

    return df


def calculate_top3_underwriters(df):
    """
    각 IPO의 상장연도 기준 전년도 Top 3 주선인 여부를 인코딩합니다.
    
    Args:
        df: '상장주선인/' 또는 '상장주선인' 컬럼과 '상장일' 컬럼을 포함한 DataFrame
    
    Returns:
        Series: Top 3 주선인이면 1, 아니면 0
    """
    # 상장주선인 컬럼 탐색
    underwriter_col = None
    for col in ['상장주선인/', '상장주선인']:
        if col in df.columns:
            underwriter_col = col
            break
    
    if underwriter_col is None:
        print("  [Warning] 상장주선인 컬럼 없음 → 0으로 채움")
        return pd.Series(0, index=df.index)
    
    # 상장연도 추출
    listing_dates = pd.to_datetime(df['상장일'], errors='coerce')
    years = listing_dates.dt.year
    
    # 연도별 주선인 빈도 집계 (쉼표 구분 분리)
    year_underwriter_counts = {}
    for idx, row in df.iterrows():
        year = years.loc[idx]
        raw = str(row[underwriter_col]).strip()
        if pd.isna(year) or raw in ('nan', '', 'None'):
            continue
        # 쉼표로 구분된 복수 주선인 처리
        underwriters = [u.strip() for u in raw.split(',') if u.strip()]
        for u in underwriters:
            year_underwriter_counts.setdefault(int(year), {})
            year_underwriter_counts[int(year)][u] = year_underwriter_counts[int(year)].get(u, 0) + 1
    
    # 연도별 Top 3 주선인 산출
    top3_by_year = {}
    for year, counts in year_underwriter_counts.items():
        sorted_underwriters = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top3_by_year[year] = set(name for name, _ in sorted_underwriters[:3])
    
    # 각 행에 대해 전년도 Top 3 여부 인코딩
    result = pd.Series(0, index=df.index)
    for idx, row in df.iterrows():
        year = years.loc[idx]
        if pd.isna(year):
            continue
        prev_year = int(year) - 1
        if prev_year not in top3_by_year:
            continue
        raw = str(row[underwriter_col]).strip()
        if raw in ('nan', '', 'None'):
            continue
        underwriters = [u.strip() for u in raw.split(',') if u.strip()]
        if any(u in top3_by_year[prev_year] for u in underwriters):
            result.loc[idx] = 1
    
    print(f"  Top3 주선인 해당: {int(result.sum())}건 / {len(result)}건")
    return result


def preprocess_features(df):
    """
    피처 전처리 및 스케일링
    
    Args:
        df: 전처리할 데이터프레임
        scaler_type: 스케일러 종류 ('standard', 'robust', 'minmax')
    """
    print("\nPreprocessing Features...")
    
    # 시초가 처리
    if '상장일 시초가' in df.columns:
        df['price_open'] = pd.to_numeric(
            df['상장일 시초가'].astype(str).str.replace(',', '').str.replace(' ', ''),
            errors='coerce'
        )
    else:
        df['price_open'] = np.nan

    # 상장일 시초가 결측 → BHAR trend 파일의 시가(T=1)로 보완
    if 'price_open' in df.columns and '시가_T1' in df.columns:
        missing_mask = df['price_open'].isna() & df['시가_T1'].notna()
        if missing_mask.any():
            df.loc[missing_mask, 'price_open'] = df.loc[missing_mask, '시가_T1']
            print(f"  상장일 시초가 보완: {missing_mask.sum()}건 (BHAR trend 시가 사용)")

    # BHAR 결측치 제거
    df_clean = df.dropna(subset=['BHAR', 'price_open'])
    print(f"Data count after dropping missing BHAR: {len(df_clean)}")
    
    if len(df_clean) == 0:
        return None, None
    
    # BHAR 기반 Y 라벨링 (Target)
    df_clean = df_clean.copy()
    df_clean['Y'] = (df_clean['BHAR'] >= 0).astype(int)
    
    # Source Task Y 라벨링 (MTL)
    source_y_cols = []
    bhar_source_cols = [c for c in df_clean.columns if c.startswith('BHAR_T')]

    # 모든 Source BHAR 컬럼의 NaN을 한 번에 제거 (일관된 샘플 집합 보장)
    if bhar_source_cols:
        before_count = len(df_clean)
        nan_per_col = {col: int(df_clean[col].isna().sum()) for col in bhar_source_cols}
        df_clean = df_clean.dropna(subset=bhar_source_cols)
        total_dropped = before_count - len(df_clean)
        if total_dropped > 0:
            print(f"  Source BHAR NaN 행 일괄 제거: {before_count} → {len(df_clean)} ({total_dropped}건)")
            for col, cnt in nan_per_col.items():
                if cnt > 0:
                    print(f"    - {col}: {cnt}건 NaN")

    for col in bhar_source_cols:
        y_col = col.replace('BHAR_', 'Y_')
        df_clean[y_col] = (df_clean[col] >= 0).astype(int)
        source_y_cols.append(y_col)
        print(f"  {y_col}: 상승={int((df_clean[y_col]==1).sum())}, 하락={int((df_clean[y_col]==0).sum())}")
    
    print(f"Class Distribution (Target):\n{df_clean['Y'].value_counts()}")
    
    # Drop 컬럼 정의
    drop_columns = [
        '종목명',
        # '종목코드',  # Optuna 최적화 시 BHAR 데이터 매핑을 위해 유지
        '상장일',  # 피처에서는 제외 (별도 저장하여 시계열 Split에 사용)
        '상장유형', '증권구분',
        '상장일 시초가', '상장일 종가',
        '시장구분',
        '상장일_dt', '상장년도',
        'price_open', 'price_day5', 'return_rate', 'Y',
        '회사명', 'Unnamed: 17',
        '코스닥_15일_수익률', # Legacy
        'price_day_n', 'R_stock', 'R_benchmark', 'BHAR', # Intermediate columns
        '시가_T1', # BHAR trend 시가 (보완용)
    ]
    # Source BHAR 및 Y 컬럼도 drop (피처에서 제외, 별도 저장)
    drop_columns.extend(bhar_source_cols)
    drop_columns.extend(source_y_cols)
    
    # 상장주선인 Top 3 인코딩
    print("Calculating Top 3 Underwriter Encoding...")
    df_clean['상장주선인_Top3'] = calculate_top3_underwriters(df_clean)
    drop_columns.extend(['상장주선인/', '상장주선인'])
    
    # 증가율 컬럼 분리
    growth_rate_columns = [
        '순이익증가율(표준재무)',
        '영업이익증가율(보고서기재)(표준재무)',
        '매출액(영업수익)증가율(표준재무)',
        'EBITDA증가율(표준재무)',
    ]
    
    print("Processing growth rate columns...")
    for col in growth_rate_columns:
        if col in df_clean.columns:
            df_clean, new_cols = split_mixed_column(df_clean, col)
            print(f"  {col} → {len(new_cols)} features")
            drop_columns.append(col)
            drop_columns.append(f"{col}_상태")
    
    # 업종 컬럼은 drop하지 않고 유지 (encode_post_split에서 처리)
    
    # 피처 선택
    feature_columns = [c for c in df_clean.columns if c not in drop_columns]
    X = df_clean[feature_columns].copy()
    
    # 업종 컬럼 보존 (encode_post_split에서 원핫인코딩 처리)
    preserve_str_cols = ['업종']
    
    # 타입 변환
    cols_to_drop = []
    for col in X.columns:
        if col in preserve_str_cols:
            continue  # 문자열 유지 (split 이후 처리)
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        elif X[col].dtype == object:
            converted = pd.to_numeric(
                X[col].astype(str).str.replace(',', '').str.replace('%', ''), 
                errors='coerce'
            )
            if converted.isna().all():
                cols_to_drop.append(col)
            else:
                X[col] = converted
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
    

    # 수치형 + 보존된 문자열 컬럼 선택
    numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    str_cols = [c for c in preserve_str_cols if c in X.columns]
    X = X[numeric_cols + str_cols]
    
    # 결측치 처리는 split 이후 train 기반으로 수행 (encode_post_split)
    missing_cols = [col for col in X.columns if X[col].isna().any()]
    if missing_cols:
        print(f"  결측치 보유 컬럼 ({len(missing_cols)}): {missing_cols}")
        print(f"  → split 이후 train 기반 대체 예정")
    
    y = df_clean['Y']

    # 상장일 (시계열 기반 Split용)
    listing_dates = df_clean['상장일'] if '상장일' in df_clean.columns else None

    # Source Task Y 라벨도 반환
    y_source = {}
    for col in source_y_cols:
        if col in df_clean.columns:
            y_source[col] = df_clean[col]

    print(f"\nFeatures ({len(X.columns)}): {X.columns.tolist()}")

    # 스케일링은 train/test split 이후에 적용 (데이터 누수 방지)
    # src.utils.scale_features()를 model_trainer / benchmark에서 호출
    print(f"\n[Info] 스케일링은 train/test split 이후 적용됩니다 (data leakage 방지).")

    return X, y, y_source, listing_dates


def main():
    print("=" * 60)
    print("전처리 시작")
    print("=" * 60)
    
    output_dir = 'output/preprocess'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 데이터 로드
    finance_path = str(FINANCE_DATA_PATH)

    df = load_finance_data(finance_path)

    # 2. BHAR 데이터 로드 (3_bhar_trend.py 출력 활용, API 재호출 없음)
    priced_df = load_bhar_from_trend(df)

    # 3. 피처 전처리
    X, y, y_source, listing_dates = preprocess_features(priced_df)

    if X is None:
        print("전처리 실패")
        return

    # 4. 저장
    preprocessed_df = X.copy()
    preprocessed_df['Y'] = y.values

    # 상장일 저장 (시계열 기반 Train/Test Split용)
    if listing_dates is not None:
        preprocessed_df['상장일'] = listing_dates.values
        print(f"  상장일 컬럼 추가 (시계열 Split용)")

    # 상장일 기준 정렬
    if '상장일' in preprocessed_df.columns:
        preprocessed_df = preprocessed_df.sort_values(by='상장일').reset_index(drop=True)
        print("  Data sorted by 상장일")

    # Source Task Y 컬럼 추가 (MTL)
    for col, vals in y_source.items():
        preprocessed_df[col] = vals.values
        print(f"  {col} 추가: 상승={int((vals==1).sum())}, 하락={int((vals==0).sum())}")

    output_path = os.path.join(output_dir, 'preprocessed_data.csv')
    preprocessed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n저장 완료: {output_path}")
    print(f"  샘플 수: {len(preprocessed_df)}")
    print(f"  피처 수: {len(X.columns)}")
    print(f"  Y 분포: 상승({(y==1).sum()}), 하락({(y==0).sum()})")
    if y_source:
        print(f"  Source Y 컬럼: {list(y_source.keys())}")


if __name__ == "__main__":
    main()

