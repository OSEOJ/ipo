"""
Model Factory
중앙 집중식 모델 생성 로직
"""
import os
from ..config import (
    MLP_PARAMS,
    MMOE_PARAMS, PLE_PARAMS,
    XGBOOST_PARAMS, CATBOOST_PARAMS, LOGISTIC_PARAMS
)

def create_model(model_type, source_days=None, verbose=True, **kwargs):
    """
    모델 객체 생성 (Factory Pattern)
    
    Args:
        model_type (str): 모델 타입 ('aitm', 'mmoe', 'xgboost' 등)
        source_days (list): Source Task 날짜 리스트 (MTL 모델용)
        verbose (bool): 출력 여부
        **kwargs: 하이퍼파라미터 오버라이드 (예: Optuna 최적값)
    
    Returns:
        model: 생성된 모델 객체
    """
    model_type = model_type.lower()
    
    # 별칭 매핑
    aliases = {
        'logisticregression': 'logistic',
    }
    model_type = aliases.get(model_type, model_type)
    
    # 공통 파라미터 (정규화 포함 — 모든 DL 모델 공유)
    common_params = {
        'bottom_mlp_dims': MLP_PARAMS.get('hidden_dims', [128, 64]),
        'tower_mlp_dims': MLP_PARAMS.get('tower_dims', [64]),
        'dropout': MLP_PARAMS.get('dropout', 0.2),
        'learning_rate': MLP_PARAMS.get('learning_rate', 1e-3),
        'batch_size': MLP_PARAMS.get('batch_size', 256),
        'epochs': MLP_PARAMS.get('epochs', 100),
        'source_days': source_days or [],
        'verbose': verbose,
        'weight_decay': MLP_PARAMS.get('weight_decay', 0.0),
        'focal_gamma': MLP_PARAMS.get('focal_gamma', 0.0),
        'source_loss_weight': MLP_PARAMS.get('source_loss_weight', 0.5),
        'early_stopping_patience': None,
        'analyze_conflict': False,
        'device': None,
    }
    
    # kwargs로 넘어온 파라미터가 있으면 덮어쓰기 (DL 모델용)
    for k, v in kwargs.items():
        if k in common_params:
            common_params[k] = v
        # 구조적 파라미터 매핑 (Optuna -> Model)
        if k == 'n_layers' or k == 'h1' or k == 'h2':
            # Note: 9_benchmark.py에서 처리해서 bottom_mlp_dims로 넘기는 게 더 좋음.
            # 여기서는 단순 덮어쓰기만 허용.
            pass
    
    # --- Deep Learning (MTL) ---
    if model_type == 'aitm':
        from .aitm import AITMClassifier
        if verbose: print(f"Using AITM Model (Source days: {source_days})")
        return AITMClassifier(**common_params)
    
    elif model_type == 'aitm_seq':
        from .aitm import AITMSeqClassifier
        if verbose: print(f"Using Original AITM (Sequential) Model (Source days: {source_days})")
        return AITMSeqClassifier(**common_params)
    
    elif model_type == 'mmoe':
        from .mmoe import MMoEClassifier
        if verbose: print(f"Using MMoE Model (Source days: {source_days}, Experts: {MMOE_PARAMS['expert_num']})")
        return MMoEClassifier(
            **common_params,
            expert_num=MMOE_PARAMS.get('expert_num', 4),
        )
    
    elif model_type == 'ple':
        from .ple import PLEClassifier
        if verbose: print(f"Using PLE Model (Source days: {source_days})")
        return PLEClassifier(
            **common_params,
            shared_expert_num=PLE_PARAMS.get('shared_expert_num', 2),
            specific_expert_num=PLE_PARAMS.get('specific_expert_num', 2),
        )
    
    elif model_type == 'singletask':
        from .singletask import SingleTaskClassifier
        if verbose: print(f"Using SingleTask Model (Source days: {source_days})")
        return SingleTaskClassifier(**common_params)
    
    # --- Machine Learning (Single) ---
    elif model_type == 'xgboost':
        from .ml_baselines import XGBoostClassifier
        if verbose: print("Using XGBoost Model")
        return XGBoostClassifier(**XGBOOST_PARAMS, verbose=verbose)
    
    elif model_type == 'catboost':
        from .ml_baselines import CatBoostClassifierWrapper
        if verbose: print("Using CatBoost Model")
        return CatBoostClassifierWrapper(**CATBOOST_PARAMS, verbose=verbose)
    
    elif model_type == 'logistic':
        from .ml_baselines import LogisticRegressionClassifier
        if verbose: print("Using Logistic Regression Model")
        return LogisticRegressionClassifier(**LOGISTIC_PARAMS, verbose=verbose)
    
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")
