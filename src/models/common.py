"""
DL 모델 공통 모듈
MLP, Permutation Importance, BaseMTLClassifier를 공유합니다.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


# Gradient Clipping 기본값
DEFAULT_MAX_GRAD_NORM = 1.0


class MLP(nn.Module):
    """Multi-Layer Perceptron with BatchNorm"""

    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(p=dropout))
            input_dim = h_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def calc_permutation_importance(model, X, y, device, n_repeats=10):
    """
    Permutation Importance (Target Task 기준, AUC-ROC 기반).
    반드시 test 데이터로 호출해야 편향 없는 중요도를 얻습니다.

    Args:
        model: nn.Module (forward → list of task outputs, 마지막이 Target)
        X: numpy array (test features)
        y: numpy array (test labels)
        device: torch device
        n_repeats: 셔플 반복 횟수

    Returns:
        numpy array: 정규화된 피처 중요도
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    y_np = y

    with torch.no_grad():
        outputs = model(X_t)
        base_proba = outputs[-1].cpu().numpy()
    base_auc = roc_auc_score(y_np, base_proba)

    imps = []
    for i in range(X.shape[1]):
        orig = X_t[:, i].clone()
        drop = 0
        for _ in range(n_repeats):
            X_t[:, i] = orig[torch.randperm(len(orig))]
            with torch.no_grad():
                outputs = model(X_t)
                proba = outputs[-1].cpu().numpy()
            drop += (base_auc - roc_auc_score(y_np, proba))
        X_t[:, i] = orig
        imps.append(drop / n_repeats)

    importances = np.array(imps)
    s = importances.sum()
    if s > 0:
        importances /= s
    return importances


class BaseMTLClassifier(BaseEstimator, ClassifierMixin):
    """
    MTL 모델 공통 Base 클래스.
    weight_decay, focal_gamma, gradient clipping을 통일합니다.
    서브클래스는 _build_model()과 model_name을 구현해야 합니다.
    """

    def __init__(self, bottom_mlp_dims=None, tower_mlp_dims=None,
                 dropout=0.2, learning_rate=1e-3, batch_size=256,
                 epochs=100, source_days=None, device=None, verbose=True,
                 weight_decay=0.0, focal_gamma=0.0,
                 source_loss_weight=0.5, early_stopping_patience=None,
                 analyze_conflict=False):
        self.bottom_mlp_dims = bottom_mlp_dims or [256, 128]
        self.tower_mlp_dims = tower_mlp_dims or [64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.source_days = source_days or []
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.source_loss_weight = source_loss_weight
        self.early_stopping_patience = early_stopping_patience
        self.analyze_conflict = analyze_conflict

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.classes_ = None
        self._feature_importances_ = None
        self.training_history = []
        # Gradient conflict analysis
        self.gradient_cosine_history = []
        self.gradient_vectors_history = []
        self._task_keys = None

    # --- 서브클래스 구현 필수 ---
    model_name = 'base'

    def _build_model(self, d_in, n_classes, n_tasks):
        """nn.Module 반환. 서브클래스에서 오버라이드."""
        raise NotImplementedError

    # --- 공통 학습 루프 ---
    def fit(self, X, y_dict, X_valid=None, y_valid_dict=None, trial=None):
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        if not isinstance(y_dict, dict):
            y_dict = {'target': y_dict}

        source_keys = sorted([k for k in y_dict if k != 'target'])
        task_keys = source_keys + ['target']
        n_tasks = len(task_keys)

        y_arrays = {}
        for key, y in y_dict.items():
            y_arrays[key] = y.values if hasattr(y, 'values') else np.array(y)

        self.classes_ = np.unique(y_arrays['target'])
        n_classes = len(self.classes_)
        d_in = X.shape[1]

        # 모델 초기화 (서브클래스)
        self.model = self._build_model(d_in, n_classes, n_tasks).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # LR Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6,
        )


        # DataLoader
        tensors = [torch.FloatTensor(X)]
        for key in task_keys:
            tensors.append(torch.FloatTensor(y_arrays[key].astype(np.float32)))

        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation 데이터 준비 (선택사항)
        has_val = False
        if X_valid is not None and y_valid_dict is not None:
            has_val = True
            X_v = X_valid.values if hasattr(X_valid, 'values') else X_valid
            X_v = np.asarray(X_v, dtype=np.float32)
            X_valid_t = torch.FloatTensor(X_v).to(self.device)
            y_v = y_valid_dict.get('target', y_valid_dict) if isinstance(y_valid_dict, dict) else y_valid_dict
            y_v = y_v.values if hasattr(y_v, 'values') else np.array(y_v)
            y_valid_target_t = torch.FloatTensor(y_v.astype(np.float32)).to(self.device)
        else:
            if self.verbose:
                print("  [Info] Validation data not provided. Training on full set without EarlyStopping.")

        # 학습 루프
        self.model.train()
        self.training_history = []

        # Gradient Conflict Analysis 초기화 (AITM 전용)
        self.gradient_cosine_history = []
        self.gradient_vectors_history = []
        self._task_keys = task_keys

        # Conflict Analysis 대상: 공유 파라미터 (attention, layer_norm)
        _shared_params = [
            p for n, p in self.model.named_parameters()
            if 'attention' in n or 'layer_norm' in n
        ]

        def compute_task_gradients_safe(batch_X, batch_ys):
            """
            각 Task별 공유 파라미터 gradient 계산.
            torch.autograd.grad()를 사용하여 param.grad 오염 없이 안전하게 계산.
            """
            if not _shared_params:
                return {}

            task_grads = {}
            with torch.enable_grad():
                outputs = self.model(batch_X)
                for i, key in enumerate(task_keys):
                    y_pred = outputs[i].clamp(1e-7, 1 - 1e-7)
                    task_loss = nn.functional.binary_cross_entropy(
                        y_pred, batch_ys[key]
                    )
                    grads = torch.autograd.grad(
                        task_loss, _shared_params,
                        retain_graph=True, allow_unused=True,
                    )
                    grad_list = [g.flatten() for g in grads if g is not None]
                    if grad_list:
                        task_grads[key] = torch.cat(grad_list).detach().cpu().numpy()
            return task_grads

        def compute_cosine_similarities(task_grads):
            """Task 쌍 간 cosine similarity 계산"""
            cos_sims = {}
            keys = list(task_grads.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    g1, g2 = task_grads[keys[i]], task_grads[keys[j]]
                    dot = np.dot(g1, g2)
                    norm = np.linalg.norm(g1) * np.linalg.norm(g2)
                    cos_sim = dot / (norm + 1e-8)
                    cos_sims[(keys[i], keys[j])] = cos_sim
            return cos_sims

        # Source task 개별 weight 계산 (상수)
        # source_loss_weight는 전체 source 예산 (source 수와 무관하게 총합 고정)
        n_source = len(source_keys)
        per_source_weight = self.source_loss_weight / max(n_source, 1)

        # Early Stopping 상태
        es_best_loss = float('inf')
        es_best_epoch = 0
        es_best_state = None

        for epoch in range(self.epochs):
            epoch_losses = {key: 0.0 for key in task_keys}
            n_batches = 0
            epoch_conflict_measured = False

            for batch in loader:
                batch_X = batch[0].to(self.device)
                batch_ys = {task_keys[i]: batch[i + 1].to(self.device) for i in range(n_tasks)}

                # Gradient Conflict Analysis (에폭당 첫 배치만, AITM 전용)
                # 학습 forward/backward 전에 별도 수행하여 gradient 간섭 방지
                if (self.analyze_conflict and self.model_name == 'AITM'
                    and not epoch_conflict_measured):
                    task_grads = compute_task_gradients_safe(batch_X, batch_ys)
                    if task_grads:
                        cos_sims = compute_cosine_similarities(task_grads)
                        self.gradient_cosine_history.append(cos_sims)
                        self.gradient_vectors_history.append(task_grads)
                    epoch_conflict_measured = True

                # 학습용 forward (conflict analysis와 분리)
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                total_loss = 0
                for i, key in enumerate(task_keys):
                    y_true = batch_ys[key]
                    y_pred = outputs[i]

                    # BCE Loss
                    # clamp로 수치 안정성 확보 (log(0) → CUDA assert 방지)
                    y_pred_safe = y_pred.clamp(1e-7, 1 - 1e-7)
                    bce = nn.functional.binary_cross_entropy(
                        y_pred_safe, y_true, reduction='none'
                    )

                    # Focal Loss
                    if self.focal_gamma > 0:
                        p_t = y_true * y_pred_safe + (1 - y_true) * (1 - y_pred_safe)
                        focal_weight = (1 - p_t) ** self.focal_gamma
                        bce = focal_weight * bce

                    task_loss = bce.mean()

                    if key == 'target':
                        total_loss += task_loss * 1.0
                    else:
                        total_loss += task_loss * per_source_weight

                    epoch_losses[key] += task_loss.item()

                total_loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=DEFAULT_MAX_GRAD_NORM
                )

                optimizer.step()
                n_batches += 1

            avg_losses = {key: epoch_losses[key] / max(n_batches, 1) for key in task_keys}
            self.training_history.append({'epoch': epoch + 1, 'losses': avg_losses})

            # Validation loss 계산 (선택사항)
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_valid_t)
                    val_pred = val_outputs[-1].clamp(1e-7, 1 - 1e-7)  # Target task
                    monitor_loss = nn.functional.binary_cross_entropy(
                        val_pred, y_valid_target_t
                    ).item()
                self.model.train()
            else:
                monitor_loss = avg_losses['target']

            # LR Scheduler (validation loss 기반, 없으면 target train loss)
            scheduler.step(monitor_loss)

            # Optuna Pruning (Validation 값이 있을 때만)
            if trial is not None and has_val:
                import optuna
                trial.report(-monitor_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Early Stopping (Validation 값이 있을 때만)
            if self.early_stopping_patience is not None and has_val:
                if monitor_loss < es_best_loss:
                    es_best_loss = monitor_loss
                    es_best_epoch = epoch
                    es_best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                elif epoch - es_best_epoch >= self.early_stopping_patience:
                    self.model.load_state_dict(es_best_state)
                    if self.verbose:
                        print(f"  [EarlyStopping] Stopped at epoch {epoch + 1}, "
                              f"best at {es_best_epoch + 1} (val_loss={es_best_loss:.4f})")
                    break

            if self.verbose and (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                loss_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
                if has_val:
                    loss_str += f", val={monitor_loss:.4f}"
                print(f"  Epoch [{epoch + 1}/{self.epochs}] {loss_str} | lr={current_lr:.2e}")

        # Early Stopping 사용 시 best 가중치 복원 (break 없이 종료된 경우)
        if self.early_stopping_patience is not None and es_best_state is not None:
            self.model.load_state_dict(es_best_state)

        print(f"[{self.model_name}] Training complete ({epoch + 1}/{self.epochs} epochs)")
        return self

    # --- 공통 predict ---
    @property
    def feature_importances_(self):
        return self._feature_importances_

    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """Target Task(마지막 Task) 확률만 반환"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            target_proba = outputs[-1].cpu().numpy()
        return np.column_stack([1 - target_proba, target_proba])
