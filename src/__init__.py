"""
Multimodal Recommender System using LVLMs
"""

from .dataset import MultimodalDataset
from .recommender import TitleOnlyRecommender, LVLMEnhancedRecommender
from .metrics import recall_at_k, ndcg_at_k, hit_rate_at_k, evaluate_model, print_evaluation_results

__version__ = "0.1.0"
__all__ = [
    "MultimodalDataset",
    "TitleOnlyRecommender",
    "LVLMEnhancedRecommender",
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "evaluate_model",
    "print_evaluation_results",
]
