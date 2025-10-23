"""
Prepare data for training and testing. 
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Kaggle multimodal dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dataset", type=str, help="Override dataset name from config")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(archive_path: Path, dataset_name: str):
    dataset_path = archive_path / dataset_name
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    
    train_df = pd.read_csv(dataset_path / "train.txt")
    val_df = pd.read_csv(dataset_path / "validation.txt")
    test_df = pd.read_csv(dataset_path / "test.txt")
    
    items_df = pd.read_csv(dataset_path / "items.txt")
    users_df = pd.read_csv(dataset_path / "users.txt")
    
    image_embeds = np.load(dataset_path / "embed_image.npy")
    text_embeds = np.load(dataset_path / "embed_text.npy")
    
    print(f"  Train: {len(train_df)} interactions")
    print(f"  Val: {len(val_df)} interactions")
    print(f"  Test: {len(test_df)} interactions")
    print(f"  Users: {len(users_df)}")
    print(f"  Items: {len(items_df)}")
    print(f"  Image embeds: {image_embeds.shape}")
    print(f"  Text embeds: {text_embeds.shape}")
    
    return train_df, val_df, test_df, items_df, users_df, image_embeds, text_embeds


def save_processed_data(output_path: Path, train_df, val_df, test_df, items_df, users_df, image_embeds, text_embeds):
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    items_df.to_csv(output_path / "items.csv", index=False)
    users_df.to_csv(output_path / "users.csv", index=False)
    
    np.save(output_path / "image_embeds.npy", image_embeds)
    np.save(output_path / "text_embeds.npy", text_embeds)
    
    print(f"\nSaved processed data to {output_path}")


def main():
    args = parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    dataset_name = args.dataset or config['data']['dataset']
    archive_path = Path(config['data']['archive_path'])
    processed_path = Path(config['data']['processed_path'])
    
    print("="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    train_df, val_df, test_df, items_df, users_df, image_embeds, text_embeds = load_dataset(archive_path, dataset_name)
    
    save_processed_data(processed_path, train_df, val_df, test_df, items_df, users_df, image_embeds, text_embeds)
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
