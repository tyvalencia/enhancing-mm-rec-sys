"""
Models package for multimodal recommendation.
"""

from .attention_fusion import AttentionFusionRecommender
from .base_model import BaseRecommender
from .smore import SMORERecommender

__all__ = ['AttentionFusionRecommender', 'BaseRecommender', 'SMORERecommender']

