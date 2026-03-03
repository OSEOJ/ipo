"""
IPO 크롤링 스크립트
38.co.kr에서 IPO 기본 정보를 수집하여 CSV로 저장합니다.
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_START_DATE, DEFAULT_END_DATE
from src.utils import parse_date
from src.crawlers.ipo_crawler import IPOCrawler


def main(start=None, end=None, limit=None, output=None):
    # 키워드 인자가 없으면 argparse로 CLI 파싱
    if all(v is None for v in [start, end, limit, output]):
        parser = argparse.ArgumentParser(
            description="IPO 기본 정보 크롤링",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
예시:
  # config.py의 기본 날짜로 크롤링
  python scripts/crawl.py

  # 날짜 지정
  python scripts/crawl.py --start 2024-01-01 --end 2024-12-31

  # 개수 제한
  python scripts/crawl.py --limit 10

  # 출력 파일 지정
  python scripts/crawl.py --output data/my_ipo_list.csv
            """
        )

        parser.add_argument("--start", type=str, help=f"시작일 (YYYY-MM-DD) [기본값: {DEFAULT_START_DATE}]")
        parser.add_argument("--end", type=str, help="종료일 (YYYY-MM-DD) [기본값: 오늘 날짜]")
        parser.add_argument("--limit", type=int, help="최대 수집 개수")
        parser.add_argument("--output", type=str, help="출력 CSV 경로 [기본값: output/ipo_crawl.csv]")

        args = parser.parse_args()
        start, end, limit, output = args.start, args.end, args.limit, args.output

    # 날짜 파싱
    start_dt = parse_date(start) or parse_date(DEFAULT_START_DATE)
    end_dt = parse_date(end) or parse_date(DEFAULT_END_DATE) or datetime.now()

    print("\n" + "=" * 60)
    print("IPO 기본 정보 크롤링")
    print("=" * 60)
    print(f"기간: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    if limit:
        print(f"수집 제한: {limit}개")

    # 크롤링
    crawler = IPOCrawler()
    ipo_list = crawler.crawl(start_date=start_dt, end_date=end_dt, limit=limit)

    df = pd.DataFrame(ipo_list)
    print(f"\n총 {len(df)}개 종목 수집 완료")
    print(f"컬럼: {', '.join(df.columns.tolist())}")

    # 저장
    if output:
        output_path = Path(output)
    else:
        output_path = Path("output") / "crawl" / "ipo_crawl.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
