"""
Generate LVLM summaries for products.
"""

import argparse
import json
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LVLM summaries")
    parser.add_argument("--model", type=str, default="dummy", 
                       choices=["gemini", "gpt4v", "dummy"])
    parser.add_argument("--max-items", type=int, default=None, 
                       help="Max items to process (default: all items)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_product_metadata(metadata_path: Path):
    """Load matched product metadata if available."""
    if not metadata_path.exists():
        print(f"Warning: No metadata file found at {metadata_path}")
        print("Run: python scripts/download_amazon_metadata.py")
        print("Then: python scripts/match_metadata.py")
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_product_prompt(product_meta: dict, prompt_template: str) -> str:
    title = product_meta.get('title', '')
    description = product_meta.get('description', '')
    features = product_meta.get('features', [])
    brand = product_meta.get('brand', '')
    
    # Build rich product context
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
    if brand:
        context_parts.append(f"Brand: {brand}")
    if features:
        features_text = "; ".join(features[:5])  # Top 5 features
        context_parts.append(f"Features: {features_text}")
    if description:
        # Limit description length
        desc = description[:500] if len(description) > 500 else description
        context_parts.append(f"Description: {desc}")
    
    product_context = "\n".join(context_parts)
    
    # Use template or create default
    if '{item_id}' in prompt_template:
        # Replace generic prompt with rich context
        return prompt_template.replace('{item_id}', f"\n{product_context}")
    else:
        return f"{prompt_template}\n\n{product_context}"


# Currently not optimal due to rate limits  
def generate_gemini_summaries(items_df: pd.DataFrame, prompt_template: str, 
                              product_metadata: dict = None, max_items: int = None) -> dict:
    num_items = len(items_df) if max_items is None else min(len(items_df), max_items)
    print(f"Generating Gemini summaries for {num_items} items...")
    print("⏱️  Rate limiting enabled: 6s delay between requests (~10 RPM)")
    
    import google.generativeai as genai
    import os
    import time
    
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in your environment")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    summaries = {}
    for idx in tqdm(range(num_items)):
        row = items_df.iloc[idx]
        item_id = row.get('item_id', idx)
        
        # Use real metadata if available
        if product_metadata and str(idx) in product_metadata:
            prompt = create_product_prompt(product_metadata[str(idx)], prompt_template)
        else:
            # Fallback to simple prompt
            prompt = prompt_template.format(item_id=item_id)
        
        try:
            response = model.generate_content(prompt)
            summaries[int(idx)] = response.text
            time.sleep(6)  # Rate limit: 10 requests/min max
        except Exception as e:
            # Check if it's a quota error (429)
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"\n⚠️  Hit API limit at item {idx}")
                print(f"✓ Successfully generated {len([s for s in summaries.values() if len(s) > 50])} real summaries")
                break  # Stop trying if we hit quota
            print(f"Error processing item {idx}: {e}")
            summaries[int(idx)] = f"Product {item_id}"
    
    return summaries

# Extra data if we need more data for Gemini
def generate_dummy_summaries(items_df: pd.DataFrame, max_items: int = None) -> dict:
    num_items = len(items_df) if max_items is None else min(len(items_df), max_items)
    print(f"Generating dummy summaries for {num_items} items...")
    
    # Varied templates for diversity
    templates = [
        "Product {item_id}: Premium quality item with excellent craftsmanship and durability.",
        "Item {item_id} features innovative design and outstanding performance for everyday use.",
        "High-quality {item_id} offering great value, comfort, and reliability for customers.",
        "Product {item_id} combines style and functionality with superior materials and construction.",
        "{item_id} is a versatile item designed for maximum convenience and user satisfaction.",
        "Exceptional product {item_id} with unique features and modern aesthetic appeal.",
        "{item_id}: Durable and practical item perfect for various applications and needs.",
        "Quality {item_id} delivering reliable performance and long-lasting value to users."
    ]
    
    summaries = {}
    for idx in tqdm(range(num_items)):
        item_id = items_df.iloc[idx]['item_id'] if 'item_id' in items_df.columns else f"ITEM{idx}"
        # Use different template for variety
        template = templates[idx % len(templates)]
        summaries[int(idx)] = template.format(item_id=item_id)
    
    return summaries


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    prompt_template = config['lvlm']['prompt']
    
    print("="*50)
    print("LVLM SUMMARY GENERATION")
    print("="*50)
    print(f"Prompt: {prompt_template}")
    print()
    
    items_file = Path("data/processed/items.csv")
    items_df = pd.read_csv(items_file)
    print(f"Loaded {len(items_df)} items")
    
    # Load product metadata (if available)
    metadata_file = Path("data/processed/items_with_metadata.json")
    product_metadata = load_product_metadata(metadata_file)
    
    if product_metadata:
        print(f"✓ Using real Amazon product data for {len(product_metadata)} items")
    else:
        print("⚠ No metadata found - using generic prompts")
    
    print()
    
    if args.model == "gemini":
        summaries = generate_gemini_summaries(items_df, prompt_template, product_metadata, args.max_items)
    else: 
        summaries = generate_dummy_summaries(items_df, args.max_items)
    
    output_file = Path("data/summaries/item_summaries.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\n✓ Saved {len(summaries)} summaries to {output_file}")


if __name__ == "__main__":
    main()
