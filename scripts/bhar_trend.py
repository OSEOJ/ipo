
import logging
import sys
from collections import Counter

import pandas as pd
import numpy as np
import os
import yfinance as yf
import FinanceDataReader as fdr
from datetime import timedelta
import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FINANCE_DATA_PATH, KOSPI_SYMBOL, KOSDAQ_SYMBOL

def load_ipo_list():
    """Load IPO list from data file (ipo_crawl.csv) with SPAC filtering"""
    data_path = os.path.join('output', 'crawl', 'ipo_crawl.csv')
    print(f"Loading IPO Data from {data_path}...")

    if not os.path.exists(data_path):
        # Fallback to config path if ipo_crawl.csv doesn't exist
        print(f"  [Warning] {data_path} not found. Using default FINANCE_DATA_PATH.")
        data_path = FINANCE_DATA_PATH

    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # Ensure correct types
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)

    # Filter valid markets
    if '시장구분' in df.columns:
        df = df[df['시장구분'].isin(['KOSPI', 'KOSDAQ'])]

    # Filter SPACs (referencing 3_preprocess.py logic)
    # 3_preprocess.py: merged_df = merged_df[~merged_df['종목명'].str.contains('스팩', na=False)]
    if '종목명' in df.columns:
        original_len = len(df)
        df = df[~df['종목명'].str.contains('스팩', na=False)]
        print(f"Filtered SPACs: {original_len} -> {len(df)}")

    print(f"Loaded {len(df)} IPOs.")
    return df

def get_market_data(start_date, end_date):
    """Fetch market index data (KOSPI/KOSDAQ)"""
    market_data = {}
    print("Fetching Market Index Data...")
    for sym in [KOSPI_SYMBOL, KOSDAQ_SYMBOL]:
        try:
            # Fetch with buffer
            df = yf.download(f'^{sym}', start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            market_data[sym] = df
            print(f"  {sym}: {len(df)} rows")
        except Exception as e:
            print(f"  Error fetching {sym}: {e}")
    return market_data

def get_bhar_trend(df, market_data, horizon=160):
    """
    Calculate BHAR trend for each IPO for T=1 to horizon.
    Returns a long-format DataFrame.

    O(N*T) 증분 계산으로 최적화.
    """
    results = []
    skip_reasons = Counter()

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing IPOs"):
        name = row.get('종목명', row.get('회사명', 'Unknown'))
        code = row['종목코드']
        listing_date_str = row['상장일']
        market_type = row.get('시장구분', 'KOSDAQ')

        try:
            listing_date = pd.to_datetime(listing_date_str)
        except (ValueError, TypeError):
            logger.warning(f"[{code} {name}] 상장일 파싱 실패: '{listing_date_str}'")
            skip_reasons['상장일_파싱_실패'] += 1
            continue

        # Determine Market Symbol
        market_symbol = 'KS11' if market_type == 'KOSPI' else 'KQ11'
        if market_symbol not in market_data:
            logger.warning(f"[{code} {name}] 시장지수 데이터 없음: {market_symbol}")
            skip_reasons['시장지수_없음'] += 1
            continue

        df_mkt = market_data[market_symbol]

        # Fetch Stock Data
        end_date_fetch = listing_date + timedelta(days=365)
        yf_ticker = f"{code}.KS" if market_type == 'KOSPI' else f"{code}.KQ"

        try:
            df_stock = yf.download(yf_ticker, start=listing_date, end=end_date_fetch, progress=False)
            if df_stock is None or len(df_stock) == 0:
                df_stock = fdr.DataReader(code, start=listing_date, end=end_date_fetch)

            if isinstance(df_stock.columns, pd.MultiIndex):
                df_stock.columns = df_stock.columns.get_level_values(0)

            if len(df_stock) < 1:
                logger.warning(f"[{code} {name}] 주가 데이터 없음 (0건)")
                skip_reasons['주가_데이터_없음'] += 1
                continue

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"[{code} {name}] 주가 다운로드 실패: {type(e).__name__}: {e}")
            skip_reasons['주가_다운로드_실패'] += 1
            continue

        # Base Prices (D0 Open)
        try:
            stock_open_0 = float(df_stock.iloc[0]['Open'])
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.warning(f"[{code} {name}] D0 시가 추출 실패: {type(e).__name__}: {e}")
            skip_reasons['D0_시가_추출_실패'] += 1
            continue

        if stock_open_0 <= 0:
            logger.warning(f"[{code} {name}] D0 시가 비정상: {stock_open_0}")
            skip_reasons['D0_시가_비정상'] += 1
            continue

        # Market Base (D0 Open)
        mkt_after_listing = df_mkt[df_mkt.index >= listing_date]
        if len(mkt_after_listing) == 0:
            logger.warning(f"[{code} {name}] 상장일 이후 시장지수 없음")
            skip_reasons['상장후_시장지수_없음'] += 1
            continue

        market_open_0 = float(mkt_after_listing.iloc[0]['Open'])
        if market_open_0 <= 0:
            logger.warning(f"[{code} {name}] 시장지수 D0 시가 비정상: {market_open_0}")
            skip_reasons['시장지수_D0_비정상'] += 1
            continue

        # --- 시장 데이터를 주식 거래일에 맞춰 한 번에 reindex (O(T) 전처리) ---
        max_len = min(len(df_stock), horizon)
        stock_dates = df_stock.index[:max_len]
        stock_closes = df_stock['Close'].iloc[:max_len].values
        stock_opens = df_stock['Open'].iloc[:max_len].values

        mkt_closes_series = df_mkt.reindex(stock_dates)['Close'].ffill()

        # --- 증분 누적 수익률 계산 (O(T)) ---
        # 초기값
        compound_stock = 1.0
        compound_market = 1.0
        prev_stock_close = None
        prev_mkt_close = None

        for t in range(max_len):
            stock_close_t = float(stock_closes[t])
            stock_open_t = float(stock_opens[t])
            current_date = stock_dates[t]

            # 시장 데이터 확인
            if pd.isna(mkt_closes_series.iloc[t]):
                continue
            market_close_t = float(mkt_closes_series.iloc[t])

            # BHAR_S (Simple Price Difference) - 직접 계산
            r_stock_simple = (stock_close_t - stock_open_0) / stock_open_0
            r_bench_simple = (market_close_t - market_open_0) / market_open_0
            bhar_s = r_stock_simple - r_bench_simple

            # BHAR (Cumulative Product) - 증분 계산
            if t == 0:
                # Day 0: (Close_0 - Open_0) / Open_0
                r_s = (stock_close_t - stock_open_0) / stock_open_0
                r_m = (market_close_t - market_open_0) / market_open_0
                compound_stock = 1 + r_s
                compound_market = 1 + r_m
            else:
                # Day t: daily return from previous close
                if prev_stock_close > 0 and prev_mkt_close > 0:
                    r_s_daily = stock_close_t / prev_stock_close - 1
                    r_m_daily = market_close_t / prev_mkt_close - 1
                    compound_stock *= (1 + r_s_daily)
                    compound_market *= (1 + r_m_daily)

            prev_stock_close = stock_close_t
            prev_mkt_close = market_close_t

            bhar_product = (compound_stock - 1) - (compound_market - 1)

            results.append({
                '기업명': name,
                '종목코드': code,
                '상장일': listing_date_str,
                '시장': market_type,
                'T': t + 1,
                '종가': stock_close_t,
                '시가': stock_open_t,
                '시장지수_종가': market_close_t,
                '시장지수_시초가': market_open_0,
                'BHAR': bhar_product,
                'BHAR_S': bhar_s,
            })

    # 건너뛴 종목 요약 출력
    if skip_reasons:
        total_skipped = sum(skip_reasons.values())
        print(f"\n[Skip Summary] 총 {total_skipped}건 건너뜀:")
        for reason, count in skip_reasons.most_common():
            print(f"  - {reason}: {count}건")

    return pd.DataFrame(results)

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # 1. Load List
    df_ipo = load_ipo_list()

    # 2. Fetch Global Market Data (Optimization)
    # Range: Min Listing Date to Max Listing Date + 1 Year
    dates = pd.to_datetime(df_ipo['상장일'])
    start_date = dates.min()
    end_date = dates.max() + timedelta(days=400) # Buffer

    market_data = get_market_data(start_date, end_date)

    # 3. Process
    print(f"Calculating BHAR Trend (Horizon: 160 days)...")
    result_df = get_bhar_trend(df_ipo, market_data, horizon=160)

    # Check if we have T=160
    if not result_df.empty:
        max_t = result_df['T'].max()
        print(f"Max T processed: {max_t}")

    # 4. Save
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ipo_bhar_trend_160d.csv')

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Done. Saved to {output_path}")
    print(f"Total Rows: {len(result_df)}")

if __name__ == "__main__":
    main()
