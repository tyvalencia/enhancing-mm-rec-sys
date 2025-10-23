"""
Baseline recommender models for multimodal recommendation.
"""

import numpy as np
import torch
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender:    
    def __init__(self, device="cpu"):
        self.device = device
        self.item_embeddings = None
        self.item_ids = None
    
    def build_item_index(self, item_embeddings: np.ndarray, item_ids: List[int]):
        self.item_embeddings = item_embeddings
        self.item_ids = np.array(item_ids)
        print(f"Built index with {len(item_ids)} items, embedding dimension: {item_embeddings.shape[1]}")
    
    def recommend(self, user_history: List[int], k: int = 10, exclude_history: bool = True) -> List[int]:
        # Compute user profile, which is a mean of history embeddings
        history_embeddings = self.item_embeddings[user_history]
        user_profile = history_embeddings.mean(axis=0).reshape(1, -1)
        
        # Cosine similarity
        similarities = cosine_similarity(user_profile, self.item_embeddings)[0]
        
        # If requested, exclude history items
        if exclude_history:
            similarities[user_history] = -np.inf
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return self.item_ids[top_k_indices].tolist()


class TitleOnlyRecommender(BaseRecommender):    
    def __init__(self, device="cpu"):
        super().__init__(device)
        print("Initialized TitleOnlyRecommender (text embeddings only)")


class LVLMEnhancedRecommender(BaseRecommender):
    def __init__(self, device="cpu", encoder_model="all-MiniLM-L6-v2"):
        super().__init__(device)
        self.encoder_model = encoder_model
        self.encoder = None
        print(f"Initialized LVLMEnhancedRecommender (encoder: {encoder_model})")
    
    def _load_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer # Lazy load sentence transformer
            self.encoder = SentenceTransformer(self.encoder_model, device=self.device)
            print(f"Loaded encoder: {self.encoder_model}")
    
    def build_item_index(self, summaries: Dict[int, str], item_ids: List[int]):
        self.item_ids = np.array(item_ids)
        
        missing = [item_id for item_id in item_ids if item_id not in summaries] # missing summaries check
        if missing:
            raise ValueError(f"Missing summaries for {len(missing)} items.")
        
        print(f"Encoding {len(item_ids)} LVLM summaries with Sentence-BERT...")
        self._load_encoder()
        
        # Encode all summaries
        summary_texts = [summaries[item_id] for item_id in item_ids]
        self.item_embeddings = self.encoder.encode(
            summary_texts, 
            convert_to_numpy=True, 
            show_progress_bar=True, 
            batch_size=128 
)
        
        print(f"Built LVLM index with {len(item_ids)} items, embedding dim: {self.item_embeddings.shape[1]}")


class SMOREFusionRecommender(BaseRecommender):    
    def __init__(self, device="cpu", text_weight=0.5, image_weight=0.5):
        super().__init__(device)
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.text_embeddings = None
        self.image_embeddings = None
        print(f"Initialized SMOREFusionRecommender (text_w={text_weight}, image_w={image_weight})")
    
    def build_item_index(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray, item_ids: List[int]):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.item_ids = np.array(item_ids)
        
        # Smore fusion 
        self.item_embeddings = (self.text_weight * text_embeddings + self.image_weight * image_embeddings)
        
        # Normalize
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        self.item_embeddings = self.item_embeddings / (norms + 1e-8)
        
        print(f"Built SMORE index with {len(item_ids)} items")
        print(f"  Text embeds: {text_embeddings.shape}")
        print(f"  Image embeds: {image_embeddings.shape}")
        print(f"  Fused embeds: {self.item_embeddings.shape}")
