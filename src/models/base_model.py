"""
Base recommender class for all models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


class BaseRecommender(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.item_embeddings = None
        self.item_ids = None
        
    def build_item_index(self, embeddings: np.ndarray, item_ids: List[int]):
        self.item_embeddings = torch.FloatTensor(embeddings).to(self.device)
        self.item_ids = item_ids
        
        self.item_embeddings = nn.functional.normalize(
            self.item_embeddings, p=2, dim=1
        )
    
    def get_user_profile(self, user_history: List[int]) -> torch.Tensor:
        indices = [self.item_ids.index(item_id) for item_id in user_history 
                   if item_id in self.item_ids]
        
        if not indices:
            return torch.zeros(1, self.item_embeddings.shape[1]).to(self.device)
        
        history_embeddings = self.item_embeddings[indices]
        user_profile = history_embeddings.mean(dim=0, keepdim=True)
        
        user_profile = nn.functional.normalize(user_profile, p=2, dim=1)
        
        return user_profile
    
    def recommend(self, user_history: List[int], k: int = 10, exclude_history: bool = True) -> List[int]:
        user_profile = self.get_user_profile(user_history)
        
        similarities = torch.matmul(user_profile, self.item_embeddings.t()).squeeze()
        
        if exclude_history:
            for item_id in user_history:
                if item_id in self.item_ids:
                    idx = self.item_ids.index(item_id)
                    similarities[idx] = -float('inf')
        
        _, top_indices = torch.topk(similarities, k)
        top_items = [self.item_ids[idx] for idx in top_indices.cpu().numpy()]
        
        return top_items
    
    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'item_ids': self.item_ids
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.item_ids = checkpoint['item_ids']
