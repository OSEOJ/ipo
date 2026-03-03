"""
군집별 T=22 IPO 주가성과 비교

attention_analysis_results.json의 cluster 할당 + preprocessed_data.csv의 Y 라벨을
비교하여 어느 attention 군집이 22일 IPO 성과가 가장 좋은지 확인합니다.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

# ── 경로 ──────────────────────────────────────────────────────────────
JSON_PATH  = 'output/attention_analysis_results.json'
DATA_PATH  = 'output/preprocess/preprocessed_data.csv'
TEST_SIZE  = 0.2
# ──────────────────────────────────────────────────────────────────────

# 1. JSON에서 sample_id → cluster 매핑 로드
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    results = json.load(f)

points = results['visualization_data']['scatter_data']['points']
cluster_map = {p['sample_id']: p['cluster'] for p in points}
n_test = len(cluster_map)
print(f"[JSON] 군집 할당 샘플 수: {n_test}개")

# 2. 전처리 데이터 로드 → 동일한 시계열 분할로 test set Y 추출
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
df['상장일'] = pd.to_datetime(df['상장일'], errors='coerce')
df = df[df['상장일'].notna()].copy()
df = df.sort_values('상장일').reset_index(drop=True)

n_total = len(df)
n_split = int(n_total * TEST_SIZE)
df_test = df.iloc[n_total - n_split:].reset_index(drop=True)

print(f"[Data] 전체 {n_total}건 → 테스트셋 {len(df_test)}건 "
      f"({df_test['상장일'].min().date()} ~ {df_test['상장일'].max().date()})")

if len(df_test) != n_test:
    print(f"[Warning] 샘플 수 불일치: JSON={n_test}, CSV={len(df_test)}")

# 3. cluster 할당 붙이기
df_test['cluster'] = df_test.index.map(cluster_map)

# 4. 군집별 성과 집계
source_names = ['T=1', 'T=68', 'T=149']
print("\n" + "=" * 55)
print("  군집별 T=22 IPO 주가 상승 비율 비교")
print("=" * 55)
print(f"  {'군집':<10} {'N':>5}  {'상승':>5}  {'하락':>5}  {'상승률':>8}")
print("-" * 55)

perf = {}
for name in source_names:
    subset = df_test[df_test['cluster'] == name]['Y']
    n       = len(subset)
    rise    = int(subset.sum())
    fall    = n - rise
    rate    = rise / n if n > 0 else 0.0
    perf[name] = {'n': n, 'rise': rise, 'fall': fall, 'rate': rate}
    print(f"  {name:<10} {n:>5}  {rise:>5}  {fall:>5}  {rate:>8.1%}")

print("=" * 55)

# 전체 base rate
y_all   = df_test['Y']
overall = y_all.mean()
print(f"\n  전체 상승률 (baseline): {overall:.1%}  (N={len(y_all)})")

# 5. 카이제곱 검정 (군집 간 성과 차이 유의성)
contingency = [[perf[name]['rise'], perf[name]['fall']] for name in source_names]
chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
print(f"\n  Chi-square test: χ²={chi2:.4f}, p={p_val:.4f} (dof={dof})")
if p_val < 0.05:
    print("  → 군집 간 성과 차이 통계적으로 유의함 (p<0.05)")
else:
    print("  → 군집 간 성과 차이 통계적으로 유의하지 않음 (p≥0.05)")

# 6. 최고 성과 군집
best = max(perf, key=lambda k: perf[k]['rate'])
print(f"\n  최고 성과 군집: {best}  (상승률 {perf[best]['rate']:.1%})")
