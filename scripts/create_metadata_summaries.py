#!/usr/bin/env python3
"""
Create product summaries from Amazon metadata (no LLM needed).
Uses concatenated title + description + features.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create summaries from Amazon metadata")
    parser.add_argument("--items-path", type=str, default="data/processed/items.csv")
    parser.add_argument("--metadata-path", type=str, default="data/processed/items_with_metadata.json")
    parser.add_argument("--output-path", type=str, default="data/summaries/item_summaries.json")
    parser.add_argument("--max-items", type=int, default=None, help="Max items to process (for testing)")
    return parser.parse_args()


def create_summary_from_metadata(metadata: dict) -> str:
    parts = []

    if metadata.get('title'):
        parts.append(metadata['title'])

    if metadata.get('brand'):
        parts.append(f"Brand: {metadata['brand']}")

    features = metadata.get('features', [])
    if features:
        feature_text = ". ".join(features[:5])
        parts.append(feature_text)
    
    if metadata.get('description'):
        desc = metadata['description']
        if isinstance(desc, list):
            desc = " ".join(desc)
        if len(desc) > 200:
            desc = desc[:200] + "..."
        parts.append(desc)
    
    return ". ".join(parts)


def main():
    args = parse_args()
    
    print("="*60)
    print("METADATA SUMMARY GENERATOR")
    print("="*60)
    
    items_df = pd.read_csv(args.items_path)
    print(f"\n✓ Loaded {len(items_df)} items")
    
    with open(args.metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    print(f"√ Loaded metadata for {len(metadata_dict)} products")
    
    summaries = {}
    num_items = len(items_df) if args.max_items is None else min(len(items_df), args.max_items)
    
    print(f"\nCreating summaries for {num_items} items...")
    
    for idx in tqdm(range(num_items)):
        if str(idx) in metadata_dict: # use rich metadata
            summary = create_summary_from_metadata(metadata_dict[str(idx)])
        else:
            item_row = items_df.iloc[idx]
            summary = item_row.get('title', f'Item {idx}')
        
        summaries[int(idx)] = summary
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\n✓ Saved {len(summaries)} summaries to {output_path}")
    

if __name__ == "__main__":
    main()