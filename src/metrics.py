"""
Evaluation metrics for recommendation systems
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    if not actual:
        return 0.0
    
    predicted_k = set(predicted[:k])
    actual_set = set(actual)
    
    hits = len(predicted_k.intersection(actual_set))
    recall = hits / len(actual_set)
    
    return recall


def ndcg_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    if not actual:
        return 0.0
    
    predicted_k = predicted[:k]
    actual_set = set(actual)
    
    dcg = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(len(actual), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


def hit_rate_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    if not actual:
        return 0.0
    
    predicted_k = set(predicted[:k])
    actual_set = set(actual)
    
    if len(predicted_k.intersection(actual_set)) > 0:
        return 1.0
    else:
        return 0.0


def average_precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    if not actual:
        return 0.0
    
    predicted_k = predicted[:k]
    actual_set = set(actual)
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(predicted_k):
        if item in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    if not num_hits:
        return 0.0
    
    return score / min(len(actual), k)


def evaluate_model(recommendations: Dict[str, List[int]], ground_truth: Dict[str, List[int]],
    k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    results = defaultdict(list)
    
    for user_id in ground_truth:
        if user_id not in recommendations:
            continue
            
        actual = ground_truth[user_id]
        predicted = recommendations[user_id]
        
        for k in k_values:
            results[f'recall@{k}'].append(recall_at_k(actual, predicted, k))
            results[f'ndcg@{k}'].append(ndcg_at_k(actual, predicted, k))
            results[f'hit_rate@{k}'].append(hit_rate_at_k(actual, predicted, k))
    
    avg_results = {
        metric: np.mean(scores) if scores else 0.0
        for metric, scores in results.items()
    }
    
    return avg_results


def print_evaluation_results(results: Dict[str, float]):
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    k_values = sorted(set(int(k.split('@')[1]) for k in results.keys()))
    
    for k in k_values:
        print(f"\n--- Top-{k} Metrics ---")
        for metric, score in results.items():
            if f'@{k}' in metric:
                print(f"  {metric:20s}: {score:.4f}")
    
    print("="*50 + "\n")
