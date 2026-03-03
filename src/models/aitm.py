"""
Parallel Information Transfer Multi-task (PITM) Model

본 모델은 AITM(Xi et al., KDD 2021)의 표현 수준 전이(Representation-level transfer)
아이디어를 참조하되, IPO 가격 경로가 결정적 순차 구조를 따르지 않는다는 점을 고려하여
단조 제약을 제거한 병렬 다중과제 구조를 설계하였다.

Original AITM Reference:
    Xi, Dongbo, et al. "Modeling the sequential dependence among audience
    multi-step conversions with multi-task learning in targeted display advertising."
    KDD 2021.

변형 사항:
    - 순차 전이(Task k → Task k+1) 제거 → 병렬 전이(각 Task ↔ 나머지 전체)
    - 모든 Task가 동등하게 다른 Task들의 표현을 Attention으로 참조
    - Task 배치 순서에 의한 정보 비대칭 해소
"""
import numpy as np
import torch
import torch.nn as nn

from .common import MLP, BaseMTLClassifier


class AITMModel(nn.Module):
    """
    Parallel Information Transfer Multi-task Model

    구조:
    1. 각 Task별 독립 Bottom MLP로 표현 추출
    2. 병렬 Cross-Attention: 각 Task가 나머지 모든 Task를 Attend
       - Query: 자기 자신의 표현
       - Key/Value: 나머지 Task들의 표현
    3. Residual Connection + LayerNorm
    4. Task별 Tower MLP로 최종 예측

    Original AITM과의 차이:
    - AITM: Task k는 fea[:k] (이전 Task들만) 참조 → 순차적, 비대칭
    - PITM: Task k는 fea[≠k] (나머지 전체) 참조 → 병렬적, 대칭
    """

    def __init__(self, d_in, n_classes, bottom_mlp_dims, tower_mlp_dims,
                 task_num, dropout=0.2, num_heads=None):
        super().__init__()
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]

        # num_heads 자동 결정: hidden_dim의 약수 중 최대 4까지
        if num_heads is None:
            for h in [4, 2, 1]:
                if self.hidden_dim % h == 0:
                    num_heads = h
                    break

        # 각 Task별 독립 Bottom MLP
        self.bottom = nn.ModuleList([
            MLP(d_in, bottom_mlp_dims, dropout, output_layer=False)
            for _ in range(task_num)
        ])

        # Multi-Head Attention (Parallel Information Transfer)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(bottom_mlp_dims[-1])

        # Task별 Tower MLP
        self.tower = nn.ModuleList([
            MLP(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
            for _ in range(task_num)
        ])

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: Float tensor (batch_size, d_in)
            return_attention_weights: True이면 attention weights도 반환
        Returns:
            list of tensors: 각 Task의 sigmoid 확률 (batch_size,)
            (optional) dict: {task_idx: attn_weights (B, 1, task_num-1)}
        """
        # 각 Task별 Bottom MLP 통과
        fea = [self.bottom[i](x) for i in range(self.task_num)]

        # 병렬 Information Transfer: 각 Task가 나머지 전체를 Attend
        fea_updated = []
        attn_weights_dict = {}
        for i in range(self.task_num):
            # Query: 현재 Task 표현
            curr_fea = fea[i].unsqueeze(1)  # (B, 1, H)

            # Key/Value: 나머지 모든 Task의 표현
            other_feas = torch.stack(
                [fea[j] for j in range(self.task_num) if j != i], dim=1
            )  # (B, task_num-1, H)

            # Cross-Attention
            attn_output, attn_w = self.attention(
                query=curr_fea,
                key=other_feas,
                value=other_feas
            )

            if return_attention_weights:
                attn_weights_dict[i] = attn_w  # (B, 1, task_num-1)

            # Residual Connection & LayerNorm
            fea_updated.append(
                self.layer_norm(fea[i] + attn_output.squeeze(1))
            )

        # Tower MLP를 통한 최종 예측
        results = [
            torch.sigmoid(self.tower[i](fea_updated[i]).squeeze(1))
            for i in range(self.task_num)
        ]

        if return_attention_weights:
            return results, attn_weights_dict
        return results


class AITMClassifier(BaseMTLClassifier):
    """
    PITM sklearn 호환 래퍼

    Task 배치:
    - Source Tasks를 앞에, Target Task를 맨 뒤에 배치
    - 병렬 구조이므로 순서에 의한 정보 비대칭 없음
    - predict/predict_proba는 Target Task(마지막) 결과만 반환
    """
    model_name = 'AITM'

    def _build_model(self, d_in, n_classes, n_tasks):
        return AITMModel(
            d_in=d_in,
            n_classes=n_classes,
            bottom_mlp_dims=self.bottom_mlp_dims,
            tower_mlp_dims=self.tower_mlp_dims,
            task_num=n_tasks,
            dropout=self.dropout,
        )

    def extract_attention_weights(self, X):
        """
        Target Task의 Source Task별 Attention Weight 추출.

        Args:
            X: numpy array 또는 DataFrame (N, d_in)

        Returns:
            attn_weights: numpy array (N, n_source_tasks)
                각 열은 source task에 대한 attention weight.
                Task 배치 순서: [source_0, source_1, ...] (self.source_days 순서)
            source_names: list of str (예: ['T=1', 'T=68', 'T=149'])
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs, attn_weights_dict = self.model(X_t, return_attention_weights=True)

        # Target task는 마지막 (index = task_num - 1)
        target_idx = self.model.task_num - 1
        target_attn = attn_weights_dict[target_idx]  # (B, 1, n_sources)
        target_attn = target_attn.squeeze(1).cpu().numpy()  # (B, n_sources)

        source_names = [f'T={d}' for d in self.source_days]
        return target_attn, source_names

    def save_checkpoint(self, path, feature_names):
        """
        모델 체크포인트 저장.

        Args:
            path: 저장 경로 (.pt)
            feature_names: 학습에 사용된 피처 이름 리스트
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'd_in': self.model.bottom[0].mlp[0].in_features,
                'n_classes': len(self.classes_) if self.classes_ is not None else 2,
                'n_tasks': self.model.task_num,
                'bottom_mlp_dims': self.bottom_mlp_dims,
                'tower_mlp_dims': self.tower_mlp_dims,
                'dropout': self.dropout,
            },
            'source_days': self.source_days,
            'feature_names': list(feature_names),
            'classes_': self.classes_.tolist() if self.classes_ is not None else [0, 1],
        }
        torch.save(checkpoint, path)
        print(f"[AITM] Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, path, device=None):
        """
        체크포인트에서 모델 복원.

        Args:
            path: 체크포인트 경로 (.pt)
            device: torch device (None이면 자동 감지)

        Returns:
            model: 복원된 AITMClassifier
            feature_names: 학습에 사용된 피처 이름 리스트
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint['model_config']

        model = cls(
            bottom_mlp_dims=cfg['bottom_mlp_dims'],
            tower_mlp_dims=cfg['tower_mlp_dims'],
            dropout=cfg['dropout'],
            source_days=checkpoint['source_days'],
            device=device,
        )
        model.classes_ = np.array(checkpoint['classes_'])

        # 내부 nn.Module 생성 및 가중치 로드
        model.model = model._build_model(
            d_in=cfg['d_in'],
            n_classes=cfg['n_classes'],
            n_tasks=cfg['n_tasks'],
        ).to(device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.eval()

        feature_names = checkpoint['feature_names']
        print(f"[AITM] Checkpoint loaded: {path} ({cfg['n_tasks']} tasks, {cfg['d_in']} features)")
        return model, feature_names


class AITMSeqModel(nn.Module):
    """
    Original Sequential Information Transfer Multi-task Model (Baseline)

    구조 (단조 제약은 제거함):
    1. 각 Task별 독립 Bottom MLP로 표현 추출
    2. 순차적 Cross-Attention:
       Task k는 Task 0부터 Task k-1까지의 표현만 Attend
    3. Residual Connection + LayerNorm
    4. Task별 Tower MLP로 최종 예측
    """

    def __init__(self, d_in, n_classes, bottom_mlp_dims, tower_mlp_dims,
                 task_num, dropout=0.2, num_heads=None):
        super().__init__()
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]

        # num_heads 자동 결정: hidden_dim의 약수 중 최대 4까지
        if num_heads is None:
            for h in [4, 2, 1]:
                if self.hidden_dim % h == 0:
                    num_heads = h
                    break

        # 각 Task별 독립 Bottom MLP
        self.bottom = nn.ModuleList([
            MLP(d_in, bottom_mlp_dims, dropout, output_layer=False)
            for _ in range(task_num)
        ])

        # Multi-Head Attention (Sequential Transfer)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(bottom_mlp_dims[-1])

        # Task별 Tower MLP
        self.tower = nn.ModuleList([
            MLP(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
            for _ in range(task_num)
        ])

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: Float tensor (batch_size, d_in)
            return_attention_weights: True이면 attention weights도 반환
        Returns:
            list of tensors: 각 Task의 sigmoid 확률 (batch_size,)
            (optional) dict: {task_idx: attn_weights (B, 1, i)} (task 0은 없음)
        """
        # 각 Task별 Bottom MLP 통과
        fea = [self.bottom[i](x) for i in range(self.task_num)]

        # 순차적 Information Transfer: Task k는 이전 Task들(0 ~ k-1)만 참조
        fea_updated = []
        attn_weights_dict = {}
        for i in range(self.task_num):
            if i == 0:
                # 첫 번째 Task (T=1)는 이전 Task가 없으므로 자기 자신 통과
                fea_updated.append(fea[0])
            else:
                # Query: 현재 Task 표현
                curr_fea = fea[i].unsqueeze(1)  # (B, 1, H)

                # Key/Value: 이전 Task들의 표현 (0부터 i-1까지)
                # 원본 fea 리스트에서 가져옴
                prev_feas = torch.stack(
                    [fea[j] for j in range(i)], dim=1
                )  # (B, i, H)

                # Cross-Attention
                attn_output, attn_w = self.attention(
                    query=curr_fea,
                    key=prev_feas,
                    value=prev_feas
                )

                if return_attention_weights:
                    attn_weights_dict[i] = attn_w  # (B, 1, i)

                # Residual Connection & LayerNorm
                fea_updated.append(
                    self.layer_norm(fea[i] + attn_output.squeeze(1))
                )

        # Tower MLP를 통한 최종 예측
        results = [
            torch.sigmoid(self.tower[i](fea_updated[i]).squeeze(1))
            for i in range(self.task_num)
        ]

        if return_attention_weights:
            return results, attn_weights_dict
        return results


class AITMSeqClassifier(BaseMTLClassifier):
    """
    Original AITM Sequential Baseline sklearn 호환 래퍼
    Target이 맨 마지막인 Option 1 강제 사용용.
    """
    model_name = 'AITM_Seq'

    def _build_model(self, d_in, n_classes, n_tasks):
        return AITMSeqModel(
            d_in=d_in,
            n_classes=n_classes,
            bottom_mlp_dims=self.bottom_mlp_dims,
            tower_mlp_dims=self.tower_mlp_dims,
            task_num=n_tasks,
            dropout=self.dropout,
        )

