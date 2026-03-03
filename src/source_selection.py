"""
Source Task 선정 로직 모듈
Optuna 등 외부 스크립트에서 재사용하기 위해 분리됨.
"""
import pandas as pd
import numpy as np
import os

def load_bhar_trend(path='output/ipo_bhar_trend_160d.csv'):
    """BHAR 추세 데이터 로드 및 피벗"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BHAR 추세 데이터가 없습니다: {path}\n"
            f"먼저 'python 0_bhar_trend.py'를 실행하세요."
        )
    
    df = pd.read_csv(path, low_memory=False)
    
    # 필수 컬럼 확인
    # BHAR_S가 있을 수도 있으므로, BHAR 컬럼 이름 확인
    bhar_col = None
    for candidate in ['BHAR', 'BHAR_S']:
        if candidate in df.columns:
            bhar_col = candidate
            break
    
    if bhar_col is None:
        raise ValueError(f"BHAR 컬럼을 찾을 수 없습니다. 컬럼: {df.columns.tolist()}")
    
    if 'T' not in df.columns:
        raise ValueError(f"'T' 컬럼이 없습니다. 컬럼: {df.columns.tolist()}")
    
    # 피벗: 행=종목코드, 열=T(일수), 값=BHAR
    pivot = df.pivot_table(index='종목코드', columns='T', values=bhar_col, aggfunc='mean')
    
    # 결측치가 너무 많은 종목 제거 (50% 이상 데이터 있는 종목만)
    min_valid = pivot.shape[1] * 0.5
    pivot = pivot.dropna(thresh=int(min_valid))
    
    return pivot


def compute_correlations(pivot, target_day):
    """Target day와 각 T의 Pearson 상관계수 계산"""
    if target_day not in pivot.columns:
        raise ValueError(f"Target day T={target_day}이 데이터에 없습니다.")
    
    target_series = pivot[target_day]
    
    correlations = {}
    for t in pivot.columns:
        if t == target_day:
            continue
        
        # 공통 유효 데이터가 있는 샘플만
        valid_mask = target_series.notna() & pivot[t].notna()
        n_valid = valid_mask.sum()
        
        if n_valid < 10:  # 최소 샘플 수
            continue
        
        corr = target_series[valid_mask].corr(pivot[t][valid_mask])
        if not np.isnan(corr):
            correlations[int(t)] = {
                'correlation': float(corr),
                'r_squared': float(corr ** 2),
                'n_samples': int(n_valid),
                'information_gain': float(1 - corr ** 2)  # 새로운 정보 비율
            }
    
    return correlations


def compute_pairwise_r_squared(pivot, days):
    """선택된 날짜들 간의 pairwise R² 계산"""
    n = len(days)
    r2_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                r2_matrix[i, j] = 1.0
            else:
                valid = pivot[days[i]].notna() & pivot[days[j]].notna()
                if valid.sum() >= 10:
                    corr = pivot[days[i]][valid].corr(pivot[days[j]][valid])
                    r2_matrix[i, j] = corr ** 2 if not np.isnan(corr) else 0
    
    return r2_matrix


def greedy_select_sources(correlations, pivot, target_day,
                          corr_min=0.4, corr_max=0.7,
                          max_pairwise_r2=0.5, max_sources=5,
                          verbose=False):
    """
    Greedy Selection으로 Source Task 선정
    """
    # 1. 후보 필터링
    candidates = []
    for t, info in correlations.items():
        abs_corr = abs(info['correlation'])
        if corr_min <= abs_corr <= corr_max:
            candidates.append({
                'day': t,
                'abs_corr': abs_corr,
                **info
            })
    
    if not candidates:
        # 범위를 넓혀서 재시도 (Optional)
        if verbose:
            print(f"[Warning] 상관계수 {corr_min}~{corr_max} 범위에 후보가 없습니다.")
        return []
    
    # 2. Information Gain 기준 정렬 (높은 순)
    candidates.sort(key=lambda x: x['information_gain'], reverse=True)
    
    if verbose:
        print(f"\n후보 Source Tasks ({len(candidates)}개):")
        for c in candidates[:5]:
            print(f"  T={c['day']:3d}: corr={c['correlation']:.3f}, R²={c['r_squared']:.3f}")
    
    # 3. Greedy Selection
    selected = []
    selected_days = []
    
    for c in candidates:
        if len(selected) >= max_sources:
            break
        
        # 이미 선택된 Source들과의 중복 확인
        is_redundant = False
        if selected_days:
            days_to_check = selected_days + [c['day']]
            r2_matrix = compute_pairwise_r_squared(
                pivot, [d for d in days_to_check]
            )
            # 새 후보와 기존 선택 간의 R² 확인
            new_idx = len(days_to_check) - 1
            for idx in range(new_idx):
                if r2_matrix[idx, new_idx] > max_pairwise_r2:
                    is_redundant = True
                    break
        
        if not is_redundant:
            selected.append(c)
            selected_days.append(c['day'])
    
    return selected


