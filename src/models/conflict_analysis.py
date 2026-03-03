"""
Task Conflict Analysis for MTL Models

Gradient Cosine Similarity 기반 Task Conflict 진단 및 시각화 모듈.
AITM 모델의 Negative Transfer 부재를 시각적으로 검증합니다.

시각화 종류:
1. Line Plot: 에폭별 Task 쌍 간 Gradient Cosine Similarity 추이
2. Heatmap: Task 쌍 간 평균 Cosine Similarity 매트릭스
3. Loss Curve: 모든 Task의 학습 Loss 추이
4. Gradient PCA: Task별 Gradient 방향 2D 투영
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib.pyplot as plt

import koreanize_matplotlib
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
from sklearn.decomposition import PCA


def plot_gradient_cosine_timeline(cos_sim_history, task_keys, save_path, suffix=''):
    """
    Line Plot: 에폭별 Gradient Cosine Similarity 추이
    
    Args:
        cos_sim_history: list of dict, 에폭별 {(task_i, task_j): cos_sim}
        task_keys: list of str, Task 이름 리스트
        save_path: str, 저장 경로
    """
    if not cos_sim_history:
        print("  [Warning] No gradient cosine similarity data to plot")
        return
        
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = list(range(1, len(cos_sim_history) + 1))
    
    # Task 쌍별로 시계열 데이터 추출
    pair_data = {}
    for epoch_idx, epoch_cos_sim in enumerate(cos_sim_history):
        for pair, value in epoch_cos_sim.items():
            if pair not in pair_data:
                pair_data[pair] = []
            pair_data[pair].append(value)
    
    plt.figure(figsize=(12, 6))
    
    # Target과의 관계만 강조 (다른 쌍은 연한 색)
    for pair, values in pair_data.items():
        task_i, task_j = pair
        label = f"{task_i} ↔ {task_j}"
        
        # Target 포함 쌍은 굵은 선
        if 'target' in (task_i, task_j):
            plt.plot(epochs[:len(values)], values, linewidth=2.5, label=label, marker='o', markersize=3)
        else:
            plt.plot(epochs[:len(values)], values, linewidth=1, alpha=0.5, label=label, linestyle='--')
    
    # Conflict 경계선 (cos_sim = 0)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Conflict Boundary')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gradient Cosine Similarity', fontsize=12)
    plt.title('Task Gradient Alignment Over Training\n(Positive = Cooperative, Negative = Conflict)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f'gradient_cosine_timeline{suffix}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filepath}")


def plot_gradient_cosine_heatmap(cos_sim_history, task_keys, save_path, suffix=''):
    """
    Heatmap: Task 쌍 간 평균 Cosine Similarity 매트릭스
    
    Args:
        cos_sim_history: list of dict, 에폭별 {(task_i, task_j): cos_sim}
        task_keys: list of str, Task 이름 리스트 (순서대로)
        save_path: str, 저장 경로
    """
    if not cos_sim_history:
        print("  [Warning] No gradient cosine similarity data to plot")
        return
        
    plt.rcParams['axes.unicode_minus'] = False
    
    n_tasks = len(task_keys)
    avg_matrix = np.zeros((n_tasks, n_tasks))
    count_matrix = np.zeros((n_tasks, n_tasks))
    
    # Task 인덱스 매핑
    task_to_idx = {task: idx for idx, task in enumerate(task_keys)}
    
    # 평균 계산
    for epoch_cos_sim in cos_sim_history:
        for (task_i, task_j), value in epoch_cos_sim.items():
            i, j = task_to_idx[task_i], task_to_idx[task_j]
            avg_matrix[i, j] += value
            avg_matrix[j, i] += value
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1
    
    # 평균
    count_matrix[count_matrix == 0] = 1
    avg_matrix /= count_matrix
    
    # 대각선은 1 (자기 자신)
    np.fill_diagonal(avg_matrix, 1.0)
    
    plt.figure(figsize=(8, 7))
    
    # Diverging colormap: 빨강(음수) - 흰색(0) - 초록(양수)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    ax = sns.heatmap(
        avg_matrix, 
        annot=True, 
        fmt='.3f',
        xticklabels=task_keys,
        yticklabels=task_keys,
        cmap=cmap,
        center=0,
        vmin=-1, 
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Avg Cosine Similarity'}
    )
    # Seaborn 내부에서 덮어쓰는 마이너스 기호 처리 강제화
    for text in ax.texts:
        if text.get_text().startswith('-'):
            text.set_text("\u002d" + text.get_text()[1:])
            
    # cbar(컬러바) 음수 틱 처리
    cbar = ax.collections[0].colorbar
    if cbar:
        # Get the current tick labels
        ticks = cbar.get_ticks()
        cbar.ax.set_yticks(ticks)
        cbar.ax.set_yticklabels([f"{t:g}".replace("-", "\u002d") for t in ticks])
        
    plt.title('Task Gradient Cosine Similarity Matrix\n(Green = Cooperative, Red = Conflict)', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f'gradient_cosine_heatmap{suffix}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filepath}")


def plot_task_loss_curves(training_history, save_path, suffix=''):
    """
    Loss Curve: 모든 Task의 학습 Loss 추이
    
    Args:
        training_history: list of dict, [{'epoch': 1, 'losses': {'source1': 0.5, ...}}, ...]
        save_path: str, 저장 경로
    """
    if not training_history:
        print("  [Warning] No training history to plot")
        return
    
    epochs = [h['epoch'] for h in training_history]
    task_keys = list(training_history[0]['losses'].keys())
    
    plt.figure(figsize=(12, 6))
    
    for task in task_keys:
        losses = [h['losses'][task] for h in training_history]
        
        # Target은 굵은 선
        if task == 'target':
            plt.plot(epochs, losses, linewidth=2.5, label=f'{task} (Target)', marker='o', markersize=3)
        else:
            plt.plot(epochs, losses, linewidth=1.5, label=task, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (BCE)', fontsize=12)
    plt.title('Multi-Task Learning Loss Curves\n(All Tasks Decreasing = No Conflict)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f'task_loss_curves{suffix}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filepath}")


def plot_gradient_pca(gradient_history, task_keys, save_path, suffix=''):
    """
    Gradient PCA: Task별 Gradient 방향 2D 투영
    
    마지막 N 에폭의 gradient를 PCA로 2D 투영하여 방향성 시각화.
    같은 방향 = 협력적, 반대 방향 = 충돌
    
    Args:
        gradient_history: list of dict, 에폭별 {task: gradient_vector (flattened)}
        task_keys: list of str, Task 이름 리스트
        save_path: str, 저장 경로
    """
    if not gradient_history:
        print("  [Warning] No gradient history for PCA")
        return
        
    plt.rcParams['axes.unicode_minus'] = False
    
    # 마지막 10개 에폭만 사용 (또는 전체가 10 미만이면 전체)
    n_epochs_to_use = min(10, len(gradient_history))
    recent_grads = gradient_history[-n_epochs_to_use:]
    
    # Task별로 gradient 수집
    task_gradients = {task: [] for task in task_keys}
    for epoch_grads in recent_grads:
        for task, grad in epoch_grads.items():
            if task in task_gradients:
                task_gradients[task].append(grad)
    
    # 평균 gradient (방향)
    avg_gradients = {}
    for task in task_keys:
        if task_gradients[task]:
            avg_grad = np.mean(task_gradients[task], axis=0)
            # 정규화 (방향만 중요)
            norm = np.linalg.norm(avg_grad)
            if norm > 1e-8:
                avg_gradients[task] = avg_grad / norm
    
    if len(avg_gradients) < 2:
        print("  [Warning] Not enough gradient data for PCA")
        return
    
    # PCA
    gradient_matrix = np.vstack(list(avg_gradients.values()))
    
    if gradient_matrix.shape[1] < 2:
        print("  [Warning] Gradient dimension too small for PCA")
        return
    
    pca = PCA(n_components=2)
    projected = pca.fit_transform(gradient_matrix)
    
    # 시각화
    plt.figure(figsize=(10, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(task_keys)))
    
    for idx, task in enumerate(avg_gradients.keys()):
        x, y = projected[idx]
        
        # 화살표 (원점에서 시작)
        arrow_scale = 0.8
        plt.arrow(0, 0, x * arrow_scale, y * arrow_scale, 
                  head_width=0.05, head_length=0.03, 
                  fc=colors[idx], ec=colors[idx], linewidth=2)
        
        # 라벨
        label_offset = 1.1
        plt.text(x * label_offset, y * label_offset, task, 
                 fontsize=12, ha='center', va='center',
                 fontweight='bold' if task == 'target' else 'normal')
    
    # 원점 표시
    plt.scatter([0], [0], c='black', s=100, zorder=5, marker='o')
    plt.text(0.02, -0.1, 'Origin', fontsize=10)
    
    # 원형 경계
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.3)
    plt.gca().add_patch(circle)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # x,y축의 음수를 강제로 기본 하이픈으로 출력하도록 포매터 적용
    import matplotlib.ticker as ticker
    formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}".replace("-", "\u002d"))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.title('Task Gradient Directions (PCA)\n(Same Direction = Cooperative, Opposite = Conflict)', fontsize=14)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f'gradient_pca{suffix}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {filepath}")


def plot_all_conflict_analysis(model, save_path='output/figures', suffix=''):
    """
    모든 Task Conflict 분석 시각화 실행

    Args:
        model: BaseMTLClassifier (AITM), gradient_conflict_history 속성 필요
        save_path: str, 저장 경로
        suffix: str, 파일명 접미사 (예: '_fold1')
    """
    os.makedirs(save_path, exist_ok=True)

    model_name = getattr(model, 'model_name', 'MTL')
    print(f"\n{'='*60}")
    print(f"  Task Conflict Analysis: {model_name}{suffix}")
    print(f"{'='*60}")

    # 1. Loss Curves (항상 가능)
    if hasattr(model, 'training_history') and model.training_history:
        plot_task_loss_curves(model.training_history, save_path, suffix=suffix)

    # 2~4. Gradient 기반 분석 (analyze_conflict=True인 경우만)
    if hasattr(model, 'gradient_cosine_history') and model.gradient_cosine_history:
        task_keys = getattr(model, '_task_keys', None)
        if task_keys is None:
            if model.training_history:
                task_keys = list(model.training_history[0]['losses'].keys())
            else:
                task_keys = ['source', 'target']

        plot_gradient_cosine_timeline(model.gradient_cosine_history, task_keys, save_path, suffix=suffix)
        plot_gradient_cosine_heatmap(model.gradient_cosine_history, task_keys, save_path, suffix=suffix)
    else:
        print("  [Info] Gradient cosine similarity not recorded. Set analyze_conflict=True in fit()")

    # 4. Gradient PCA
    if hasattr(model, 'gradient_vectors_history') and model.gradient_vectors_history:
        task_keys = getattr(model, '_task_keys', None)
        if task_keys is None and model.training_history:
            task_keys = list(model.training_history[0]['losses'].keys())
        plot_gradient_pca(model.gradient_vectors_history, task_keys, save_path, suffix=suffix)

    print(f"{'='*60}\n")


def print_conflict_summary(model):
    """
    콘솔에 Conflict 요약 출력
    
    Args:
        model: BaseMTLClassifier (AITM)
    """
    if not hasattr(model, 'gradient_cosine_history') or not model.gradient_cosine_history:
        print("  [Info] No gradient conflict data available")
        return
    
    # 전체 평균 계산
    pair_avgs = {}
    for epoch_cos_sim in model.gradient_cosine_history:
        for pair, value in epoch_cos_sim.items():
            if pair not in pair_avgs:
                pair_avgs[pair] = []
            pair_avgs[pair].append(value)
    
    print("\n" + "="*50)
    print("  Gradient Cosine Similarity Summary")
    print("="*50)
    
    has_conflict = False
    for pair, values in sorted(pair_avgs.items()):
        avg = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        
        status = "[O] Cooperative" if avg > 0 else "[X] CONFLICT"
        if min_val < 0 and avg > 0:
            status = "[!] Mixed"
        
        print(f"  {pair[0]:12s} - {pair[1]:12s}: avg={avg:+.4f} (std={std:.4f}) {status}")
        
        if avg < 0:
            has_conflict = True
    
    print("-"*50)
    if has_conflict:
        print("  [!] WARNING: Negative transfer detected!")
    else:
        print("  [O] All task pairs show positive gradient alignment")
    print("="*50 + "\n")
