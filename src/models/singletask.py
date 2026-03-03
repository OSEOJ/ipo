"""
SingleTask Model - Task별 독립 MLP (DL Baseline)

각 Task가 완전히 독립적인 네트워크를 사용.
MTL의 효과를 측정하기 위한 기준 모델.
"""
import torch
import torch.nn as nn

from .common import MLP, BaseMTLClassifier


class SingleTaskModel(nn.Module):
    """각 Task별 독립 MLP (공유 없음)"""
    
    def __init__(self, d_in, n_classes, bottom_mlp_dims, tower_mlp_dims,
                 task_num, dropout=0.2):
        super().__init__()
        self.task_num = task_num
        
        self.bottom = nn.ModuleList([
            MLP(d_in, bottom_mlp_dims, dropout, output_layer=False)
            for _ in range(task_num)
        ])
        self.tower = nn.ModuleList([
            MLP(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
            for _ in range(task_num)
        ])
    
    def forward(self, x):
        results = []
        for i in range(self.task_num):
            fea = self.bottom[i](x)
            results.append(torch.sigmoid(self.tower[i](fea).squeeze(1)))
        return results


class SingleTaskClassifier(BaseMTLClassifier):
    """SingleTask sklearn 호환 래퍼"""

    model_name = 'SingleTask'

    def _build_model(self, d_in, n_classes, n_tasks):
        return SingleTaskModel(
            d_in=d_in, n_classes=n_classes,
            bottom_mlp_dims=self.bottom_mlp_dims,
            tower_mlp_dims=self.tower_mlp_dims,
            task_num=n_tasks,
            dropout=self.dropout,
        )
