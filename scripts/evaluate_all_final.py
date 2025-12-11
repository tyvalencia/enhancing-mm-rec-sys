"""
Comprehensive evaluation of all models for final results.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Attention-based fusion model
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


# Gating mechanism for BERT + Vision
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


# Cross-modal attention for BERT + Vision
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


# Standard evaluation function for item embeddings with metrics 
def evaluate_recommender(item_embeddings, valid_items, test_df, train_df, val_df, k=10):
    item_to_idx = {item: idx for idx, item in enumerate(valid_items)}
    
    recalls = []
    ndcgs = []
    
    for user in tqdm(test_df['user'].unique(), desc="Evaluating", leave=False):
        user_train = train_df[train_df['user'] == user]['item'].values
        user_val = val_df[val_df['user'] == user]['item'].values
        user_history = np.concatenate([user_train, user_val])
        user_history = [i for i in user_history if i in item_to_idx]
        
        if len(user_history) == 0:
            continue
        
        history_indices = [item_to_idx[i] for i in user_history]
        user_emb = item_embeddings[history_indices].mean(axis=0, keepdims=True)
        
        scores = cosine_similarity(user_emb, item_embeddings)[0]
        ranked_indices = np.argsort(-scores)
        ranked_items = [valid_items[i] for i in ranked_indices]
        
        user_test = test_df[test_df['user'] == user]['item'].values
        user_test = [i for i in user_test if i in item_to_idx]
        
        if len(user_test) == 0:
            continue
        
        top_k = ranked_items[:k]
        hits = len(set(top_k) & set(user_test))
        recall = hits / len(user_test)
        recalls.append(recall)
        
        dcg = sum([1.0 / np.log2(i + 2) if ranked_items[i] in user_test else 0.0 
                   for i in range(k)])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(user_test)))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return float(np.mean(recalls)), float(np.mean(ndcgs))


# Generate fused embeddings using a trained model
def generate_fusion_embeddings(model, text_embeddings, vision_embeddings, llava_mapping):
    if isinstance(llava_mapping, dict) and 'item_ids' in llava_mapping:
        valid_items = llava_mapping['item_ids']
    else:
        valid_items = sorted([int(asin) for asin in llava_mapping.keys()])
    
    item_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for idx, asin in enumerate(tqdm(valid_items, desc="Fusing embeddings", leave=False)):
            text_emb = torch.FloatTensor(text_embeddings[idx]).unsqueeze(0).to(device)
            vision_emb = torch.FloatTensor(vision_embeddings[asin]).unsqueeze(0).to(device)
            
            fused_emb = model(text_emb, vision_emb)
            item_embeddings.append(fused_emb.cpu().numpy()[0])
    
    return np.array(item_embeddings), valid_items


def main():
    print("Model Evaluation")
    print("\n1. Loading datasets")
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    
    test_users = test['user'].unique()
    sampled_users = np.random.choice(test_users, size=max(1, len(test_users) // 100), replace=False)
    test = test[test['user'].isin(sampled_users)].reset_index(drop=True)
    print(f"Test users: {len(sampled_users)}, Test interactions: {len(test)}")
    
    print("\n2. Loading embeddings")
    bert_embeddings = np.load('data/processed/text_embeds.npy')
    llava_embeddings = np.load('data/processed/llava_next_7b_text_embeds.npy')
    
    with open('data/processed/llava_next_7b_mapping.json', 'r') as f:
        llava_mapping = json.load(f)
    
    vision_embeddings = np.load('data/processed/image_embeds.npy')
    
    items_df = pd.read_csv('data/processed/items.csv')
    all_items = items_df['item_id'].values
    
    print(f"BERT embeddings: {bert_embeddings.shape}")
    print(f"LLaVA embeddings: {llava_embeddings.shape}")
    print(f"Vision embeddings: {vision_embeddings.shape}")
    print(f"Total items: {len(all_items)}")
    
    results = {}
    
    print("Baseline Eval:")
    
    print("\nTitle-Only (BERT)")
    recall, ndcg = evaluate_recommender(bert_embeddings, all_items, test, train, val, k=10)
    results['Title-Only (BERT)'] = {'recall@10': recall, 'ndcg@10': ndcg}
    print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    print("\nVision-Only")
    recall, ndcg = evaluate_recommender(vision_embeddings, all_items, test, train, val, k=10)
    results['Vision-Only'] = {'recall@10': recall, 'ndcg@10': ndcg}
    print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    print("\nTitle+Vision (Naive 50/50)")
    naive_fusion = 0.5 * bert_embeddings + 0.5 * vision_embeddings
    recall, ndcg = evaluate_recommender(naive_fusion, all_items, test, train, val, k=10)
    results['Title+Vision (Naive 50/50)'] = {'recall@10': recall, 'ndcg@10': ndcg}
    print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    print("\nTitle+Vision (Concat)")
    concat_fusion = np.concatenate([bert_embeddings, vision_embeddings], axis=1)
    recall, ndcg = evaluate_recommender(concat_fusion, all_items, test, train, val, k=10)
    results['Title+Vision (Concat)'] = {'recall@10': recall, 'ndcg@10': ndcg}
    print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    print("LLaVA Eval:")
    
    if isinstance(llava_mapping, dict) and 'item_ids' in llava_mapping:
        llava_items = llava_mapping['item_ids']
    else:
        llava_items = sorted([int(asin) for asin in llava_mapping.keys()])
    
    print("\nLLaVA Text-Only")
    recall, ndcg = evaluate_recommender(llava_embeddings, llava_items, test, train, val, k=10)
    results['LLaVA Text-Only'] = {'recall@10': recall, 'ndcg@10': ndcg}
    if np.isnan(recall):
        print("No valid users with LLaVA coverage in test set")
    else:
        print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    print("\nLLaVA Naive Fusion: (use trained fusion models instead (Attention/Gating/Cross-Modal))")
    
    print("\nTrained Fusion Models:")
    
    fusion_models = {
        'Attention Fusion (Trained)': ('models/fusion_bert/attention_fusion.pt', AttentionFusion()),
        'Gating Fusion (Trained)': ('models/fusion_bert/gating_fusion.pt', GatingFusion()),
        'Cross-Modal Attention (Trained)': ('models/fusion_bert/cross_model_attention.pt', CrossModalAttention())
    }
    
    for idx, (model_name, (model_path, model)) in enumerate(fusion_models.items(), start=1):
        print(f"\n{model_name}")
        
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            print("Run train_fusion_bert_vision.py first!")
            results[model_name] = {'recall@10': 0.0, 'ndcg@10': 0.0, 'status': 'not_trained'}
            continue
        
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
        
        fused_embeddings, fused_items = generate_fusion_embeddings(
            model, llava_embeddings, vision_embeddings, llava_mapping
        )
        
        recall, ndcg = evaluate_recommender(fused_embeddings, fused_items, test, train, val, k=10)
        results[model_name] = {'recall@10': recall, 'ndcg@10': ndcg, 'status': 'trained'}
        print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
    
    output_path = Path('results/final_all_models.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('recall@10', 0), reverse=True)
    
    for rank, (model_name, metrics) in enumerate(sorted_results, start=1):
        status = metrics.get('status', 'evaluated')
        if status == 'not_trained' or status == 'not_implemented':
            print(f"{rank:<6} {model_name:<40} {'N/A':<12} {'N/A':<12} {status.replace('_', ' ').title():<15}")
        else:
            recall = metrics['recall@10']
            ndcg = metrics['ndcg@10']
            print(f"{rank:<6} {model_name:<40} {recall:<12.4f} {ndcg:<12.4f} {status.title():<15}")
    
    print("Finished Evaluation")


if __name__ == '__main__':
    main()