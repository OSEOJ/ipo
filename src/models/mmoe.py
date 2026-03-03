"""
MMoE (Multi-gate Mixture-of-Experts) Model

Reference:
    Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning 
    with multi-gate mixture-of-experts." KDD 2018.

핵심: 여러 Expert가 각자 다른 패턴을 학습하고,
      각 Task별 Gate가 어떤 Expert를 참고할지 가중치를 결정
"""
import torch
import torch.nn as nn

from .common import MLP, BaseMTLClassifier


class MMoEModel(nn.Module):
    """
    MMoE: Multi-gate Mixture-of-Experts
    
    구조:
    1. N개의 Expert MLP (각각 독립적으로 특징 학습)
    2. 각 Task별 Gate (Softmax -> Expert 가중 합산)
    3. Task별 Tower MLP로 최종 예측
    """
    
    def __init__(self, d_in, n_classes, bottom_mlp_dims, tower_mlp_dims,
                 task_num, expert_num, dropout=0.2):
        super().__init__()
        self.task_num = task_num
        self.expert_num = expert_num
        
        # Expert MLP들
        self.expert = nn.ModuleList([
            MLP(d_in, bottom_mlp_dims, dropout, output_layer=False)
            for _ in range(expert_num)
        ])
        
        # Task별 Gate (입력 -> Expert 가중치)
        self.gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, expert_num),
                nn.Softmax(dim=1)
            )
            for _ in range(task_num)
        ])
        
        # Task별 Tower MLP
        self.tower = nn.ModuleList([
            MLP(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
            for _ in range(task_num)
        ])
    
    def forward(self, x):
        # Gate 값 계산: 각 Task가 어떤 Expert를 얼마나 참고할지
        gate_value = [self.gate[i](x).unsqueeze(1) for i in range(self.task_num)]
        
        # Expert 출력: (batch, expert_num, hidden_dim)
        fea = torch.cat(
            [self.expert[i](x).unsqueeze(1) for i in range(self.expert_num)], 
            dim=1
        )
        
        # Gate × Expert -> Task별 특징
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        
        # Tower -> 예측
        results = [
            torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1))
            for i in range(self.task_num)
        ]
        return results


class MMoEClassifier(BaseMTLClassifier):
    """MMoE sklearn 호환 래퍼"""

    model_name = 'MMoE'

    def __init__(self, expert_num=4, **kwargs):
        super().__init__(**kwargs)
        self.expert_num = expert_num

    def _build_model(self, d_in, n_classes, n_tasks):
        return MMoEModel(
            d_in=d_in, n_classes=n_classes,
            bottom_mlp_dims=self.bottom_mlp_dims,
            tower_mlp_dims=self.tower_mlp_dims,
            task_num=n_tasks,
            expert_num=self.expert_num,
            dropout=self.dropout,
        )
