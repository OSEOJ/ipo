"""
PLE (Progressive Layered Extraction) Model

Reference:
    Tang, Hongyan, et al. "Progressive layered extraction (ple): 
    A novel multi-task learning (mtl) model for personalized recommendations."
    RecSys 2020.

핵심: 공유 Expert + Task-specific Expert를 층별(Layer-by-layer)로 점진적 추출
      MMoE보다 Task 간 간섭(Negative Transfer) 감소
"""
import torch
import torch.nn as nn

from .common import MLP, BaseMTLClassifier


class PLEModel(nn.Module):
    """
    PLE: Progressive Layered Extraction
    
    구조:
    1. 층별로:
       - Shared Expert: 모든 Task가 공유하는 Expert
       - Task-specific Expert: 각 Task 전용 Expert
       - Gate: Task별로 어떤 Expert(공유+전용)를 참고할지 결정
    2. 층이 쌓일수록 점진적으로 Task-specific한 특징 추출
    """
    
    def __init__(self, d_in, n_classes, bottom_mlp_dims, tower_mlp_dims,
                 task_num, shared_expert_num, specific_expert_num, dropout=0.2):
        super().__init__()
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)
        
        # 층별 Expert와 Gate 구성
        self.task_experts = nn.ModuleList()
        self.task_gates = nn.ModuleList()
        self.share_experts = nn.ModuleList()
        self.share_gates = nn.ModuleList()
        
        for i in range(self.layers_num):
            input_dim = d_in if i == 0 else bottom_mlp_dims[i - 1]
            
            # Shared Experts
            self.share_experts.append(nn.ModuleList([
                MLP(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)
                for _ in range(shared_expert_num)
            ]))
            
            # Shared Gate
            self.share_gates.append(nn.Sequential(
                nn.Linear(input_dim, shared_expert_num + task_num * specific_expert_num),
                nn.Softmax(dim=1)
            ))
            
            # Task-specific Experts & Gates
            layer_task_experts = nn.ModuleList()
            layer_task_gates = nn.ModuleList()
            for j in range(task_num):
                layer_task_experts.append(nn.ModuleList([
                    MLP(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)
                    for _ in range(specific_expert_num)
                ]))
                layer_task_gates.append(nn.Sequential(
                    nn.Linear(input_dim, shared_expert_num + specific_expert_num),
                    nn.Softmax(dim=1)
                ))
            self.task_experts.append(layer_task_experts)
            self.task_gates.append(layer_task_gates)
        
        # Tower MLP
        self.tower = nn.ModuleList([
            MLP(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
            for _ in range(task_num)
        ])
    
    def forward(self, x):
        # 초기 입력: 모든 Task + Shared에 동일 입력
        task_fea = [x for _ in range(self.task_num + 1)]  # +1 for shared
        
        for i in range(self.layers_num):
            # Shared Expert 출력
            share_output = [
                expert(task_fea[-1]).unsqueeze(1)
                for expert in self.share_experts[i]
            ]
            
            task_output_list = []
            for j in range(self.task_num):
                # Task-specific Expert 출력
                task_output = [
                    expert(task_fea[j]).unsqueeze(1)
                    for expert in self.task_experts[i][j]
                ]
                task_output_list.extend(task_output)
                
                # Gate: Task-specific + Shared 합쳐서 가중 합산
                mix_output = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_output).squeeze(1)
            
            # 마지막 레이어가 아니면 Shared Expert도 업데이트
            if i != self.layers_num - 1:
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_output = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_output).squeeze(1)
        
        results = [
            torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1))
            for i in range(self.task_num)
        ]
        return results


class PLEClassifier(BaseMTLClassifier):
    """PLE sklearn 호환 래퍼"""

    model_name = 'PLE'

    def __init__(self, shared_expert_num=2, specific_expert_num=2, **kwargs):
        super().__init__(**kwargs)
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num

    def _build_model(self, d_in, n_classes, n_tasks):
        return PLEModel(
            d_in=d_in, n_classes=n_classes,
            bottom_mlp_dims=self.bottom_mlp_dims,
            tower_mlp_dims=self.tower_mlp_dims,
            task_num=n_tasks,
            shared_expert_num=self.shared_expert_num,
            specific_expert_num=self.specific_expert_num,
            dropout=self.dropout,
        )
