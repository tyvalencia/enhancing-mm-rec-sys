"""
Train fusion models using BERT text (768d) + Vision (768d).
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Attention-based fusion for BERT + Vision (768d + 768d)
class AttentionFusion(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768):
        super().__init__()
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.vision_proj = nn.Linear(input_dim, hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, text_emb, vision_emb):
        text_h = self.text_proj(text_emb)
        vision_h = self.vision_proj(vision_emb)
        
        concat = torch.cat([text_h, vision_h], dim=-1)
        weights = self.attention(concat)
        
        fused = weights[:, 0:1] * text_h + weights[:, 1:2] * vision_h
        output = self.output_proj(fused)
        return output


# Gating fusion for BERT + Vision (weighs contributions of different modalities)
class GatingFusion(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768):
        super().__init__()
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.vision_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, text_emb, vision_emb):
        text_h = self.text_proj(text_emb)
        vision_h = self.vision_proj(vision_emb)
        
        concat = torch.cat([text_h, vision_h], dim=-1)
        gate_weight = self.gate(concat)
        
        fused = gate_weight * text_h + (1 - gate_weight) * vision_h
        output = self.output_proj(fused)
        return output


# Cross-modal attention for BERT + Vision (learns to weight contributions of different modalities)
class CrossModalAttention(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, num_heads=4):
        super().__init__()
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.vision_proj = nn.Linear(input_dim, hidden_dim)
        
        self.text_to_vision = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vision_to_text = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, text_emb, vision_emb):
        text_h = self.text_proj(text_emb).unsqueeze(1)
        vision_h = self.vision_proj(vision_emb).unsqueeze(1)
        
        text_att, _ = self.text_to_vision(text_h, vision_h, vision_h)
        vision_att, _ = self.vision_to_text(vision_h, text_h, text_h)
        
        fused = torch.cat([text_att.squeeze(1), vision_att.squeeze(1)], dim=-1)
        output = self.output_proj(fused)
        return output


class MultimodalDataset(Dataset):
    def __init__(self, interactions_df, text_embeddings, vision_embeddings, negative_samples=4):
        self.interactions = interactions_df
        self.text_embeddings = text_embeddings
        self.vision_embeddings = vision_embeddings
        self.negative_samples = negative_samples
        self.valid_items = list(range(len(text_embeddings)))
        
        print(f"Dataset: {len(self.interactions)} interactions, {len(self.valid_items)} items")
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        pos_item = int(row['item'])
        
        pos_text = self.text_embeddings[pos_item]
        pos_vision = self.vision_embeddings[pos_item]
        
        neg_items = np.random.choice(
            [i for i in self.valid_items if i != pos_item],
            size=self.negative_samples,
            replace=False
        )
        
        neg_texts = self.text_embeddings[neg_items]
        neg_visions = self.vision_embeddings[neg_items]
        
        return {
            'pos_text': torch.FloatTensor(pos_text),
            'pos_vision': torch.FloatTensor(pos_vision),
            'neg_texts': torch.FloatTensor(neg_texts),
            'neg_visions': torch.FloatTensor(neg_visions)
        }


# Contrastive loss needed for training fusion models
def contrastive_loss(pos_emb, neg_embs, temperature=0.07):
    pos_emb = nn.functional.normalize(pos_emb, dim=-1)
    neg_embs = nn.functional.normalize(neg_embs, dim=-1)
    
    pos_sim = torch.sum(pos_emb * pos_emb, dim=-1) / temperature
    neg_sims = torch.matmul(pos_emb.unsqueeze(1), neg_embs.transpose(1, 2)).squeeze(1) / temperature
    
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = nn.functional.cross_entropy(logits, labels)
    
    return loss


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            pos_text = batch['pos_text'].to(device)
            pos_vision = batch['pos_vision'].to(device)
            neg_texts = batch['neg_texts'].to(device)
            neg_visions = batch['neg_visions'].to(device)
            
            pos_emb = model(pos_text, pos_vision)
            
            batch_size, n_neg = neg_texts.shape[0], neg_texts.shape[1]
            neg_texts_flat = neg_texts.view(-1, neg_texts.shape[-1])
            neg_visions_flat = neg_visions.view(-1, neg_visions.shape[-1])
            neg_embs = model(neg_texts_flat, neg_visions_flat)
            neg_embs = neg_embs.view(batch_size, n_neg, -1)
            
            loss = contrastive_loss(pos_emb, neg_embs)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                pos_text = batch['pos_text'].to(device)
                pos_vision = batch['pos_vision'].to(device)
                neg_texts = batch['neg_texts'].to(device)
                neg_visions = batch['neg_visions'].to(device)
                
                pos_emb = model(pos_text, pos_vision)
                
                batch_size, n_neg = neg_texts.shape[0], neg_texts.shape[1]
                neg_texts_flat = neg_texts.view(-1, neg_texts.shape[-1])
                neg_visions_flat = neg_visions.view(-1, neg_visions.shape[-1])
                neg_embs = model(neg_texts_flat, neg_visions_flat)
                neg_embs = neg_embs.view(batch_size, n_neg, -1)
                
                loss = contrastive_loss(pos_emb, neg_embs)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            print(f"New best model")
        
        scheduler.step(avg_val_loss)
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def main():
    print("Training Fusion Models:")
    
    print("\n1. Loading data")
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    
    print(f"Train: {len(train)} interactions")
    print(f"Val: {len(val)} interactions")
    
    print("\n2. Loading embeddings")
    bert_embeddings = np.load('data/processed/text_embeds.npy')
    vision_embeddings = np.load('data/processed/image_embeds.npy')
    
    print(f"BERT embeddings: {bert_embeddings.shape}")
    print(f"Vision embeddings: {vision_embeddings.shape}")
    
    print("\n3. Creating datasets")
    train_dataset = MultimodalDataset(train, bert_embeddings, vision_embeddings)
    val_dataset = MultimodalDataset(val, bert_embeddings, vision_embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    models = {
        'Attention Fusion': AttentionFusion(),
        'Gating Fusion': GatingFusion(),
        'Cross-Modal Attention': CrossModalAttention()
    }
    
    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        
        model = model.to(device)
        model = train_model(model, train_loader, val_loader, epochs=15, lr=1e-3)
        
        model_dir = Path('models/fusion_bert')
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / f"{model_name.lower().replace(' ', '_').replace('-', '_')}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")
    
    print("\nModels saved to: models/fusion_bert/")


if __name__ == '__main__':
    main()
