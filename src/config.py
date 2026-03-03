"""
IPO 하락 예측 시스템 설정
"""
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# =========================================================
# 1. Project & Environment Settings
# =========================================================
load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")
if not DART_API_KEY:
    warnings.warn(
        "DART_API_KEY 환경변수가 설정되지 않았습니다. "
        "크롤링 실행 시 .env 파일 확인이 필요합니다.",
        stacklevel=2,
    )

OPENDART_BASE_URL = "https://opendart.fss.or.kr/api"
DART_VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# Sub-cache directories
IPO_CACHE_DIR = CACHE_DIR / "ipo"

# Ensure directories exist
for dir_path in [DATA_DIR, CACHE_DIR, IPO_CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File Paths
FINANCE_DATA_PATH = DATA_DIR / "IPO_2009_2025.csv"
MISSING_DATA_PATH = DATA_DIR / "missing_merged.xlsx"
MTL_SOURCE_TASKS_PATH = PROJECT_ROOT / "output" / "source_tasks.json"




# =========================================================
# 3. Market Data (BHAR & Returns)
# =========================================================
KOSDAQ_SYMBOL = "KQ11"
KOSPI_SYMBOL = "KS11"
MARKET_RETURN_DAYS = 15
PREDICTION_HORIZON = 22  # 예측 기간 (거래일 기준)


# =========================================================
# 4. Model General Settings
# =========================================================
MODEL_TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 43

SCALER = 'standard'

# MTL Settings
# 수동으로 Source Task 지정 (None 또는 빈 리스트면 자동 선정)
MTL_MANUAL_SOURCE_TASKS = [1, 68, 149]


# =========================================================
# 5. Model Hyperparameters (Deep Learning)
# =========================================================
# AITM, MMoE, PLE, SingleTask 공통
MLP_PARAMS = {
    'hidden_dims': [64],      # Bottom MLP
    'tower_dims': [16],           # Tower MLP
    'dropout': 0.5,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'epochs': 500,
    'weight_decay': 1e-5,
    'focal_gamma': 2.5,           # Focal Loss
    'source_loss_weight': 1.0     # Source Task Loss Weight (전체 source 예산)
}

# MMoE Specific
MMOE_PARAMS = {
    'expert_num': 4,
}

# PLE Specific
PLE_PARAMS = {
    'shared_expert_num': 2,
    'specific_expert_num': 2,
}


# =========================================================
# 6. Model Hyperparameters (Machine Learning Baselines)
# =========================================================
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
}

CATBOOST_PARAMS = {
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.1,
}

LOGISTIC_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
}


# =========================================================
# 7. Optuna Search Space
# =========================================================
OPTUNA_SEARCH_SPACE = {
    'n_layers': {'low': 1, 'high': 2},
    'h1': [64, 128, 256],
    'h2': [64, 128],
    'tower_dim': [16, 32, 64],
    'dropout': {'low': 0.3, 'high': 0.6, 'step': 0.1},
    'learning_rate': {'low': 1e-5, 'high': 1e-2, 'log': True},
    'weight_decay': {'low': 1e-6, 'high': 1e-3, 'log': True},
    'batch_size': [16, 32, 64],
    'focal_gamma': {'low': 0.5, 'high': 3.0, 'step': 0.5},
    'source_loss_weight': {'low': 0.1, 'high': 1.0, 'step': 0.3},
}
