"""
ML Baselines: XGBoost, CatBoost, Logistic Regression

전통적인 머신러닝 모델을 통일된 인터페이스로 제공.
단일 Task 전용 (MTL 미지원).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost 래퍼"""
    
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 verbose=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.classes_ = None
        self._feature_importances_ = None
    
    def fit(self, X, y, X_valid=None, y_valid=None, class_weight=None):
        from xgboost import XGBClassifier
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.classes_ = np.unique(y)
        
        # class_weight -> scale_pos_weight 변환
        scale_pos_weight = 1.0
        if class_weight and isinstance(class_weight, dict):
            scale_pos_weight = class_weight.get(1, 1.0) / class_weight.get(0, 1.0)
        
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            verbosity=0,
            use_label_encoder=False
        )
        
        if self.verbose:
            print(f"  XGBoost: n_estimators={self.n_estimators}, max_depth={self.max_depth}")
        
        self.model.fit(X, y)
        self._feature_importances_ = self.model.feature_importances_
        
        return self
    
    @property
    def feature_importances_(self):
        return self._feature_importances_
    
    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)


class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    """CatBoost 래퍼"""
    
    def __init__(self, iterations=300, depth=6, learning_rate=0.1,
                 random_state=42, verbose=True):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.classes_ = None
        self._feature_importances_ = None
    
    def fit(self, X, y, X_valid=None, y_valid=None, class_weight=None):
        from catboost import CatBoostClassifier as CBC
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.classes_ = np.unique(y)
        
        # class_weight 변환
        auto_class_weights = None
        if class_weight and isinstance(class_weight, dict):
            auto_class_weights = [class_weight.get(c, 1.0) for c in self.classes_]
        
        self.model = CBC(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            class_weights=auto_class_weights,
            verbose=0,
            allow_writing_files=False
        )
        
        if self.verbose:
            print(f"  CatBoost: iterations={self.iterations}, depth={self.depth}")
        
        self.model.fit(X, y)
        self._feature_importances_ = self.model.get_feature_importance() / 100.0
        
        return self
    
    @property
    def feature_importances_(self):
        return self._feature_importances_
    
    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict(X).flatten().astype(int)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    """Logistic Regression 래퍼"""
    
    def __init__(self, C=1.0, max_iter=1000, random_state=42, verbose=True):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.classes_ = None
        self._feature_importances_ = None
    
    def fit(self, X, y, X_valid=None, y_valid=None, class_weight=None):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.classes_ = np.unique(y)
        
        sk_class_weight = 'balanced' if class_weight else None
        
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=sk_class_weight,
            solver='lbfgs'
        )
        
        if self.verbose:
            print(f"  LogisticRegression: C={self.C}")
        
        self.model.fit(X, y)
        
        # coefficient 절대값을 feature importance로 사용
        self._feature_importances_ = np.abs(self.model.coef_[0])
        s = self._feature_importances_.sum()
        if s > 0:
            self._feature_importances_ /= s
        
        return self
    
    @property
    def feature_importances_(self):
        return self._feature_importances_
    
    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)
