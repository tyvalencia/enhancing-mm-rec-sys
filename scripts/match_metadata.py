"""
Match Amazon metadata with Kaggle dataset items.

This script matches the ASIN product IDs from the Kaggle dataset with the 
Amazon metadata to enable LVLM summary generation with real product data.
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_amazon_metadata(metadata_path: Path):
    """Load parsed Amazon metadata."""
    print(f"Loading Amazon metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata for {len(metadata)} products")
    return metadata


def match_items_with_metadata(items_df: pd.DataFrame, metadata: dict):
    """Match Kaggle items with Amazon metadata by ASIN."""
    
    print(f"\nMatching {len(items_df)} items with Amazon metadata...")
    
    matched_items = {}
    missing_count = 0
    
    for idx, row in tqdm(items_df.iterrows(), total=len(items_df)):
        # ASIN is in 'Unnamed: 0' column (the actual Amazon ASIN like B000B6AV7K)
        asin = row.get('Unnamed: 0') or row.get('item_id')
        
        if asin in metadata:
            product_meta = metadata[asin]
            
            # Create rich product description for LVLM
            matched_items[idx] = {
                'asin': asin,
                'title': product_meta.get('title', ''),
                'description': product_meta.get('description', ''),
                'features': product_meta.get('features', []),
                'brand': product_meta.get('brand', ''),
                'categories': product_meta.get('categories', [])
            }
        else:
            missing_count += 1
    
    match_rate = len(matched_items) / len(items_df) * 100
    print(f"\n✓ Matched: {len(matched_items)}/{len(items_df)} items ({match_rate:.1f}%)")
    print(f"✗ Missing: {missing_count} items")
    
    return matched_items


def save_matched_items(matched_items: dict, output_path: Path):
    """Save matched items to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(matched_items, f, indent=2)
    
    print(f"\n✓ Saved matched items to {output_path}")


def main():
    print("="*60)
    print("METADATA MATCHER")
    print("="*60)
    
    # Load Kaggle items
    items_file = Path("data/processed/items.csv")
    if not items_file.exists():
        print(f"Error: {items_file} not found. Run prepare_data.py first.")
        return
    
    items_df = pd.read_csv(items_file)
    print(f"Loaded {len(items_df)} items from Kaggle dataset")
    
    # Load Amazon metadata
    metadata_file = Path("data/raw/amazon_metadata/meta_Clothing_Shoes_and_Jewelry_parsed.json")
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found.")
        print("Run: python scripts/download_amazon_metadata.py first")
        return
    
    metadata = load_amazon_metadata(metadata_file)
    
    # Match items
    matched_items = match_items_with_metadata(items_df, metadata)
    
    # Save matched items
    output_file = Path("data/processed/items_with_metadata.json")
    save_matched_items(matched_items, output_file)
    
    print("\n" + "="*60)
    print("NEXT STEP")
    print("="*60)
    print("Run: python scripts/generate_summaries.py --model gemini")
    print("This will use real product data to generate LVLM summaries!")


if __name__ == "__main__":
    main()

