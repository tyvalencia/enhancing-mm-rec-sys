"""
Attention-based fusion model for multimodal recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from .base_model import BaseRecommender


# Multi-head attention fusion module
class AttentionFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 384,
        meta_dim: int = 384,
        img_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        text_emb: torch.Tensor,
        meta_emb: torch.Tensor,
        img_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_h = self.text_proj(text_emb)
        meta_h = self.meta_proj(meta_emb)
        img_h = self.img_proj(img_emb)
        
        features = torch.stack([text_h, meta_h, img_h], dim=1)
        
        attn_output, attn_weights = self.attention(
            features, features, features
        )
        
        fused = attn_output.mean(dim=1)
        fused = self.output_proj(fused)
        
        return fused, attn_weights


# Recommender with attention-based multimodal fusion
class AttentionFusionRecommender(BaseRecommender):
    def __init__(
        self,
        text_dim: int = 384,
        meta_dim: int = 384,
        img_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: str = "cpu",
        use_attention: bool = True
    ):
        super().__init__(device)
        
        self.text_dim = text_dim
        self.meta_dim = meta_dim
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        if use_attention:
            self.fusion = AttentionFusion(
                text_dim=text_dim,
                meta_dim=meta_dim,
                img_dim=img_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ).to(device)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(text_dim + meta_dim + img_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ).to(device)
        
        self.to(device)
    
    def build_multimodal_index(
        self,
        text_embeddings: np.ndarray,
        meta_embeddings: np.ndarray,
        img_embeddings: np.ndarray,
        item_ids: List[int]
    ):
        self.item_ids = item_ids
        
        text_emb = torch.FloatTensor(text_embeddings).to(self.device)
        meta_emb = torch.FloatTensor(meta_embeddings).to(self.device)
        img_emb = torch.FloatTensor(img_embeddings).to(self.device)
        
        with torch.no_grad():
            if self.use_attention:
                fused_emb, _ = self.fusion(text_emb, meta_emb, img_emb)
            else:
                concat = torch.cat([text_emb, meta_emb, img_emb], dim=1)
                fused_emb = self.fusion(concat)
        
        self.item_embeddings = F.normalize(fused_emb, p=2, dim=1)
    
    def get_attention_weights(
        self,
        item_id: int
    ) -> Optional[np.ndarray]:
        if not self.use_attention or item_id not in self.item_ids:
            return None
        
        idx = self.item_ids.index(item_id)
        return None


# Contrastive loss for training the fusion model
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        
        anchor_exp = anchor.unsqueeze(1)
        neg_sim = F.cosine_similarity(anchor_exp, negatives, dim=2)
        
        loss = -torch.log(
            torch.exp(pos_sim / self.temperature) /
            (torch.exp(pos_sim / self.temperature) + 
             torch.exp(neg_sim / self.temperature).sum(dim=1))
        ).mean()
        
        return loss
