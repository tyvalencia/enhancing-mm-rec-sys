"""
Dataset utilities for multimodal recommendation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


def load_kaggle_dataset(archive_path: str, dataset_name: str) -> Tuple:
    dataset_path = Path(archive_path) / dataset_name
    
    train_df = pd.read_csv(dataset_path / "train.txt")
    val_df = pd.read_csv(dataset_path / "validation.txt")
    test_df = pd.read_csv(dataset_path / "test.txt")
    items_df = pd.read_csv(dataset_path / "items.txt")
    users_df = pd.read_csv(dataset_path / "users.txt")
    
    image_embeds = np.load(dataset_path / "embed_image.npy")
    text_embeds = np.load(dataset_path / "embed_text.npy")
    
    return train_df, val_df, test_df, items_df, users_df, image_embeds, text_embeds


def build_interaction_matrix(interactions_df: pd.DataFrame, n_users: int, n_items: int):
    from scipy.sparse import csr_matrix
    
    rows = interactions_df['user'].values
    cols = interactions_df['item'].values
    data = np.ones(len(interactions_df))
    
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def get_user_history(interactions_df: pd.DataFrame, user_id: int) -> List[int]:
    user_interactions = interactions_df[interactions_df['user'] == user_id]
    return user_interactions['item'].tolist()


def get_item_embedding(item_embeds: np.ndarray, item_id: int) -> np.ndarray:
    return item_embeds[item_id]
