"""
IPO 수익률 예측 시스템 - 통합 실행 스크립트

파이프라인 순서:
  scripts/crawl.py           - IPO 크롤링
  scripts/bhar_trend.py      - BHAR 추세 계산
  scripts/preprocess.py      - 데이터 전처리
  scripts/source_tasks.py    - Source Task 최적화 + 선정
  scripts/shap_analysis.py   - SHAP 기반 피처 선택 (RFE)
  scripts/optuna_search.py   - Optuna HP 탐색 (AITM)
  scripts/train_evaluate.py  - 모델 학습 및 벤치마크 비교
"""
import argparse
import sys
from pathlib import Path


def main():
    COMMANDS = ['crawl', 'bhar', 'preprocess', 'source_task', 'shap', 'optuna', 'benchmark', 'attention']

    parser = argparse.ArgumentParser(
        description="IPO 수익률 예측 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용법:
  python main.py <명령어> [옵션]

명령어 목록:
  crawl        : IPO 크롤링
  bhar         : BHAR 추세 계산
  preprocess   : 데이터 전처리
  source_task  : Source Task 최적화 + 선정
  shap         : SHAP 피처 셀렉션
  optuna       : Optuna HP 탐색 (AITM)
  benchmark    : 모델 학습 및 벤치마크
  attention    : Attention 전이 메커니즘 분석
        """
    )

    parser.add_argument(
        "command",
        nargs='?',
        choices=COMMANDS,
        help="실행할 명령어"
    )

    try:
        from src.config import SCALER
    except ImportError:
        SCALER = 'standard'

    parser.add_argument(
        "--scaler", "-s",
        type=str,
        choices=['standard', 'robust', 'minmax'],
        default=SCALER,
        help=f"전처리 스케일러 종류 (기본값: {SCALER})"
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=60,
        help="Optuna 탐색 횟수 (기본값: 60)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default='all',
        help="벤치마크/학습 실행할 모델 (all, aitm, xgboost 등)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="학습 랜덤 시드 (벤치마크/학습 단계에서 사용)"
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="벤치마크 seed 반복 횟수 (기본값: 1)"
    )

    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="seed 증가 간격 (기본값: 1)"
    )

    parser.add_argument(
        "--ckpt-suffix",
        type=str,
        default='',
        help="AITM 체크포인트 파일 suffix (예: _seed43)"
    )
    
    parser.add_argument(
        "--no-shap",
        action='store_true',
        help="SHAP 피처 선택 비활성화 (벤치마크용)"
    )

    parser.add_argument(
        "--source-mode",
        type=str,
        choices=['optimize', 'select', 'auto'],
        default='auto',
        help="Source Task 실행 모드 (기본값: auto)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    command = args.command

    if command == "crawl":
        print("\n[1] 크롤링 실행 중...")
        from scripts.crawl import main as run_crawl
        run_crawl()

    elif command == "bhar":
        print("\n[2] BHAR 추세 계산 실행 중...")
        from scripts.bhar_trend import main as run_bhar
        run_bhar()

    elif command == "preprocess":
        print("\n[3] 전처리 실행 중...")
        from scripts.preprocess import main as run_preprocess
        run_preprocess()
    
    elif command == "source_task":
        print(f"\n[4] Source Task ({args.source_mode}) 실행 중...")
        from scripts.source_tasks import main as run_source
        run_source(mode=args.source_mode)

    elif command == "shap":
        print("\n[3] 전처리 자동 실행 (SHAP 전 선행작업)...")
        from scripts.preprocess import main as run_preprocess
        run_preprocess()

        print("\n[5] SHAP 피처 셀렉션 실행 중 (RFE Step=1)...")
        from scripts.shap_analysis import main as run_shap
        run_shap()

    elif command == "optuna":
        print("\n[6] Optuna HP 탐색 실행 중...")
        from scripts.optuna_search import main as run_optuna
        run_optuna(n_trials=args.n_trials)
        
    elif command == "benchmark":
        print(f"\n[7] 모델 학습 및 벤치마크 실행 중 (Target: {args.model})...")
        from scripts.train_evaluate import main as run_benchmark
        run_benchmark(
            model=args.model,
            no_shap=args.no_shap,
            seed=args.seed,
            n_seeds=args.n_seeds,
            seed_step=args.seed_step,
            ckpt_suffix=args.ckpt_suffix,
        )

    elif command == "attention":
        print("\n[8] Attention 전이 메커니즘 분석 실행 중...")
        from scripts.attention_analysis import main as run_attention
        run_attention()

    elif command == "all":
        print("[INFO] 'all' 명령은 현재 지원되지 않습니다. 각 단계를 순차적으로 실행해주세요. (예: python main.py crawl)")


if __name__ == "__main__":
    main()
