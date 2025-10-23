"""
Run recommendation experiment with baseline models.
"""

import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommender import TitleOnlyRecommender, LVLMEnhancedRecommender, SMOREFusionRecommender
from metrics import evaluate_model, print_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run recommendation experiment")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model-type", type=str, choices=["title_only", "lvlm_enhanced", "smore_fusion"])
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"])
    parser.add_argument("--use-validation", action="store_true", help="Evaluate on validation set instead of test")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(processed_path: Path, use_validation: bool = False):
    train_df = pd.read_csv(processed_path / "train.csv")
    eval_df = pd.read_csv(processed_path / ("val.csv" if use_validation else "test.csv"))
    items_df = pd.read_csv(processed_path / "items.csv")
    users_df = pd.read_csv(processed_path / "users.csv")
    
    image_embeds = np.load(processed_path / "image_embeds.npy")
    text_embeds = np.load(processed_path / "text_embeds.npy")
    
    eval_type = "validation" if use_validation else "test"
    print(f"Loaded {len(train_df)} train, {len(eval_df)} {eval_type} interactions")
    print(f"Items: {len(items_df)}, Users: {len(users_df)}")
    
    return train_df, eval_df, items_df, users_df, text_embeds, image_embeds


def build_user_histories(train_df: pd.DataFrame) -> Dict[int, List[int]]:
    histories = {}
    for user, group in train_df.groupby('user'):
        histories[user] = group['item'].tolist()
    return histories


def build_ground_truth(eval_df: pd.DataFrame) -> Dict[int, List[int]]:
    ground_truth = {}
    for user, group in eval_df.groupby('user'):
        ground_truth[user] = group['item'].tolist()
    return ground_truth


def load_summaries(summaries_path: Path) -> Dict[int, str]:
    summary_file = summaries_path / "item_summaries.json"
    
    if summary_file.exists():
        import json
        with open(summary_file, 'r') as f:
            summaries = json.load(f)
        # Convert string keys to ints
        return {int(k): v for k, v in summaries.items()}
    return {}


def initialize_model(model_type: str, device: str):
    if model_type == "title_only":
        return TitleOnlyRecommender(device=device)
    elif model_type == "lvlm_enhanced":
        return LVLMEnhancedRecommender(device=device)
    elif model_type == "smore_fusion":
        return SMOREFusionRecommender(device=device, text_weight=0.5, image_weight=0.5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_recommendations(model, user_histories: Dict, all_items: List[int], 
                            k: int, max_history: int) -> Dict[int, List[int]]:
    recommendations = {}
    
    for user, history in tqdm(user_histories.items(), desc="Generating recommendations"):
        recent_history = history[-max_history:] if len(history) > max_history else history
        recs = model.recommend(user_history=recent_history, k=k, exclude_history=True)
        recommendations[user] = recs
    
    return recommendations


def save_results(results: Dict, model_type: str, output_path: Path, config: dict):
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{model_type}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset: {config['data']['dataset']}\n")
        f.write(f"Device: {config['system']['device']}\n\n")
        
        for metric, score in results.items():
            f.write(f"{metric}: {score:.4f}\n")
    
    print(f"\nResults saved to {results_file}")


def main():
    args = parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Config args
    model_type = args.model_type or config['model']['type']
    device = args.device or config['system']['device']
    use_validation = args.use_validation or config['evaluation'].get('use_validation', False)
    
    print("="*60)
    print("MULTIMODAL RECOMMENDATION EXPERIMENT")
    print("="*60)
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Model: {model_type}")
    print(f"Device: {device}")
    print(f"Eval on: {'validation' if use_validation else 'test'}")
    print("="*60)
    
    # Load data
    processed_path = Path(config['data']['processed_path'])
    train_df, eval_df, items_df, users_df, text_embeds, image_embeds = \
        load_data(processed_path, use_validation)
    
    # Build histories and ground truth
    train_histories = build_user_histories(train_df)
    ground_truth = build_ground_truth(eval_df)
    
    # Filter to common users
    common_users = set(train_histories.keys()) & set(ground_truth.keys())
    train_histories = {u: train_histories[u] for u in common_users}
    ground_truth = {u: ground_truth[u] for u in common_users}
    
    print(f"\nEvaluating on {len(common_users)} users")
    
    # Initialize model
    model = initialize_model(model_type, device)
    
    # Build item index
    all_items = items_df.index.tolist()
    
    if model_type == "smore_fusion":
        model.build_item_index(text_embeds, image_embeds, all_items)
    elif model_type == "lvlm_enhanced":
        # LVLM summaries
        summaries = load_summaries(Path(config['data']['summaries_path']))
        if not summaries:
            raise ValueError("Error: Please generate LVLM summaries first.")
        model.build_item_index(summaries, all_items)
    else:
        model.build_item_index(text_embeds, all_items)
    
    # Generate recommendations
    k_values = config['evaluation']['k_values']
    max_k = max(k_values)
    max_history = config['model']['max_history_length']
    
    recommendations = generate_recommendations(model, train_histories, all_items, max_k, max_history)
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(recommendations, ground_truth, k_values)
    
    # Print and save results
    print_evaluation_results(results)
    save_results(results, model_type, Path("results"), config)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
