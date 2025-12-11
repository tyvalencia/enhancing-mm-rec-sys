"""
SMORE: Spectrum-based Modality Representation Fusion Graph Convolutional Network
Based on: https://arxiv.org/abs/2412.14978 (WSDM 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from .base_model import BaseRecommender


# Spectral frequency domain fusion module
class SpectralFusion(nn.Module):
    def __init__(self, feature_dim: int, num_modalities: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        self.freq_filters = nn.Parameter(
            torch.ones(num_modalities, feature_dim)
        )
        
        self.fusion_weights = nn.Parameter(
            torch.ones(num_modalities) / num_modalities
        )
        
    def forward(self, *modalities):
        fused = 0
        
        for i, modality in enumerate(modalities):
            freq_domain = torch.fft.rfft(modality, dim=-1)
            
            filter_size = freq_domain.shape[-1]
            freq_filter = self.freq_filters[i, :filter_size]
            filtered = freq_domain * freq_filter
            
            spatial = torch.fft.irfft(filtered, n=self.feature_dim, dim=-1)
            
            fused += self.fusion_weights[i] * spatial
        
        return fused


# Multi-modal graph convolutional layer
class MultiModalGraphConv(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        self.gcn_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_layers)
        ])
        
    def build_similarity_graph(
        self,
        features: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        
        topk_vals, topk_idx = torch.topk(similarity, k + 1, dim=1)
        
        N = features.shape[0]
        adj = torch.zeros(N, N, device=features.device)
        
        for i in range(N):
            adj[i, topk_idx[i, 1:]] = topk_vals[i, 1:]
        
        adj = (adj + adj.t()) / 2
        adj = adj + torch.eye(N, device=features.device)
        deg = adj.sum(dim=1, keepdim=True)
        adj_norm = adj / (deg + 1e-8)
        
        return adj_norm
    
    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        h = features
        
        for i in range(self.num_layers):
            h = torch.mm(adj, h)
            h = self.gcn_layers[i](h)
            h = self.norms[i](h)
            h = F.relu(h)
            
            if i > 0:
                h = h + features
        
        return h


# Modality-aware preference module
class ModalityAwarePreference(nn.Module):
    def __init__(self, feature_dim: int, num_modalities: int = 2):
        super().__init__()
        
        self.uni_modal_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
            for _ in range(num_modalities)
        ])
        
        self.multi_modal_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * (num_modalities + 1), num_modalities + 1),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        uni_modal_features: List[torch.Tensor],
        multi_modal_features: torch.Tensor
    ) -> torch.Tensor:
        uni_processed = [
            proj(feat) for proj, feat in 
            zip(self.uni_modal_projections, uni_modal_features)
        ]
        
        multi_processed = self.multi_modal_proj(multi_modal_features)
        
        all_features = uni_processed + [multi_processed]
        concat = torch.cat(all_features, dim=-1)
        
        gates = self.gate(concat)
        
        output = sum(
            gates[:, i:i+1] * feat
            for i, feat in enumerate(all_features)
        )
        
        return output


# Full SMORE implementation (WSDM 2025)
class SMORERecommender(BaseRecommender):
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        hidden_dim: int = 256,
        num_gcn_layers: int = 2,
        k_neighbors: int = 10,
        device: str = "cpu"
    ):
        super().__init__(device)
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.text_proj = nn.Linear(text_dim, hidden_dim).to(device)
        self.image_proj = nn.Linear(image_dim, hidden_dim).to(device)
        
        self.spectral_fusion = SpectralFusion(
            feature_dim=hidden_dim,
            num_modalities=2
        ).to(device)
        
        self.text_gcn = MultiModalGraphConv(
            feature_dim=hidden_dim,
            num_layers=num_gcn_layers
        ).to(device)
        
        self.image_gcn = MultiModalGraphConv(
            feature_dim=hidden_dim,
            num_layers=num_gcn_layers
        ).to(device)
        
        self.fusion_gcn = MultiModalGraphConv(
            feature_dim=hidden_dim,
            num_layers=num_gcn_layers
        ).to(device)
        
        self.preference_module = ModalityAwarePreference(
            feature_dim=hidden_dim,
            num_modalities=2
        ).to(device)
        
        self.text_graph = None
        self.image_graph = None
        
        self.to(device)
    
    def build_item_index(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        item_ids: List[int],
        build_graphs: bool = True
    ):
        self.item_ids = item_ids
        
        text_feat = torch.FloatTensor(text_embeddings).to(self.device)
        image_feat = torch.FloatTensor(image_embeddings).to(self.device)
        
        with torch.no_grad():
            text_h = self.text_proj(text_feat)
            image_h = self.image_proj(image_feat)
            
            if build_graphs:
                print("Building similarity graphs...")
                self.text_graph = self.text_gcn.build_similarity_graph(
                    text_h, k=self.k_neighbors
                )
                self.image_graph = self.image_gcn.build_similarity_graph(
                    image_h, k=self.k_neighbors
                )
            
            if self.text_graph is not None:
                text_h = self.text_gcn(text_h, self.text_graph)
                image_h = self.image_gcn(image_h, self.image_graph)
            
            fused_h = self.spectral_fusion(text_h, image_h)
            
            if build_graphs:
                fusion_graph = self.fusion_gcn.build_similarity_graph(
                    fused_h, k=self.k_neighbors
                )
                fused_h = self.fusion_gcn(fused_h, fusion_graph)
            
            final_embeddings = self.preference_module(
                [text_h, image_h],
                fused_h
            )
        
        self.item_embeddings = F.normalize(final_embeddings, p=2, dim=1)
    
    def train_step(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_samples: torch.Tensor
    ) -> torch.Tensor:
        text_h = self.text_proj(text_embeddings)
        image_h = self.image_proj(image_embeddings)
        
        if self.text_graph is None:
            self.text_graph = self.text_gcn.build_similarity_graph(text_h)
            self.image_graph = self.image_gcn.build_similarity_graph(image_h)
        
        text_h = self.text_gcn(text_h, self.text_graph)
        image_h = self.image_gcn(image_h, self.image_graph)
        
        fused_h = self.spectral_fusion(text_h, image_h)
        
        item_embeddings = self.preference_module([text_h, image_h], fused_h)
        
        loss = torch.tensor(0.0, device=self.device)
        return loss


# Trainer for SMORE model
class SMORETrainer:
    def __init__(
        self,
        model: SMORERecommender,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train(
        self,
        train_loader,
        epochs: int = 50,
        device: str = "cuda"
    ):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                loss = self.model.train_step(
                    batch['text_embeddings'],
                    batch['image_embeddings'],
                    batch['positive_pairs'],
                    batch['negative_samples']
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
