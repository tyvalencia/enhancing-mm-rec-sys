"""
Generate product summaries using LLaVA-NeXT (7B or 13B).
"""

import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import warnings
import argparse

warnings.filterwarnings('ignore')

MODEL_CONFIGS = {
    "7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "batch_size": 8, 
        "description": "LLaVA-NeXT 7B (Mistral)"
    },
    "13b": {
        "model_id": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "batch_size": 4,
        "description": "LLaVA-NeXT 13B (Vicuna)"
    }
}


def load_model(model_id, device):
    print(f"\nLoading model: {model_id}")
    
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    
    print(f"Model loaded successfully")
    print(f"Memory usage: ~{torch.cuda.memory_allocated() / 1e9:.2f}GB" if device == "cuda" else "MPS/CPU mode")
    
    return model, processor


def generate_batch_summaries(model, processor, images, titles, device):
    prompts = []
    for title in titles:
        prompt = f"""[INST] <image>\nDescribe this product in detail for an e-commerce recommendation system. 
Include: visual appearance, style, color, materials, design features, and any notable characteristics.
Product title: {title}
Provide a concise but informative description (2-3 sentences). [/INST]"""
        prompts.append(prompt)
    
    inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    summaries = processor.batch_decode(outputs, skip_special_tokens=True)
    
    clean_summaries = []
    for summary, prompt in zip(summaries, prompts):
        generated = summary.split("[/INST]")[-1].strip()
        clean_summaries.append(generated)
    
    return clean_summaries


# Calculate priority scores (popularity + recency)
def get_item_priority_scores(interactions_df):
    item_counts = interactions_df['item'].value_counts()
    item_recency = interactions_df.groupby('item')['timestamp'].max() if 'timestamp' in interactions_df.columns else None
    
    popularity_scores = (item_counts - item_counts.min()) / (item_counts.max() - item_counts.min())
    
    if item_recency is not None:
        recency_scores = (item_recency - item_recency.min()) / (item_recency.max() - item_recency.min())
        priority_scores = 0.7 * popularity_scores + 0.3 * recency_scores
    else:
        priority_scores = popularity_scores
    
    return priority_scores.sort_values(ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Generate LLaVA-NeXT summaries")
    parser.add_argument("--model", type=str, default="7b", choices=["7b", "13b"],
                        help="Model size to use (default: 7b)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/mps/cpu, default: auto-detect)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N batches (default: 10)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Maximum number of items to process (default: all)")
    args = parser.parse_args()
    
    model_config = MODEL_CONFIGS[args.model]
    model_id = model_config["model_id"]
    batch_size = model_config["batch_size"]
    
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Generating product summaries via LLaVA-NeXT ({model_config['description']})")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    print("\n1. Loading data")
    items_df = pd.read_csv('data/processed/items_with_titles.csv')
    
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        priority_scores = get_item_priority_scores(train_df)
        print(f"Prioritizing by popularity")
    except:
        priority_scores = pd.Series(1.0, index=items_df['item_id'].values)
        print(f"Using random order")
    
    image_dir = Path('data/images')
    available_images = {int(p.stem): p for p in image_dir.glob('*.jpg')}
    print(f"Available images: {len(available_images)}")
    
    items_with_images = items_df[items_df['item_id'].isin(available_images.keys())].copy()
    items_with_images['priority'] = items_with_images['item_id'].map(priority_scores).fillna(0)
    items_with_images = items_with_images.sort_values('priority', ascending=False)
    
    if args.max_items:
        items_with_images = items_with_images.head(args.max_items)
        print(f"Processing top {args.max_items} items by priority")
    
    print(f"Items to process: {len(items_with_images)}")
    
    output_file = Path(f'data/summaries/llava_next_{args.model}_summaries.json')
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_summaries = json.load(f)
        print(f"Found {len(existing_summaries)} existing summaries, will resume")
    else:
        existing_summaries = {}
    
    model, processor = load_model(model_id, device)
    
    print(f"\n2. Generating summaries (batch size: {batch_size})")
    results = existing_summaries.copy()
    
    items_to_process = [
        (row['item_id'], row.get('title', 'Unknown Product'))
        for _, row in items_with_images.iterrows()
        if str(row['item_id']) not in existing_summaries
    ]
    
    if not items_to_process:
        print("Summaries already generated")
        return
    
    for i in tqdm(range(0, len(items_to_process), batch_size), desc="Processing batches"):
        batch = items_to_process[i:i+batch_size]
        
        batch_images = []
        batch_ids = []
        batch_titles = []
        
        for item_id, title in batch:
            try:
                img_path = available_images[item_id]
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                batch_ids.append(item_id)
                batch_titles.append(title)
            except Exception as e:
                print(f"\nWarning: Failed to load {item_id}: {e}")
                continue
        
        if not batch_images:
            continue
        
        try:
            summaries = generate_batch_summaries(model, processor, batch_images, batch_titles, device)
            
            for item_id, title, summary in zip(batch_ids, batch_titles, summaries):
                results[str(item_id)] = {
                    'title': title,
                    'summary': summary,
                    'model': model_id
                }
            
            if (i // batch_size) % args.checkpoint_every == 0:
                output_file.parent.mkdir(exist_ok=True, parents=True)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"\nError processing batch {i//batch_size}: {e}")
            continue
    
    print(f"\n3. Saving results")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Total summaries: {len(results)}")
    print(f"Output file: {output_file}")
    

if __name__ == '__main__':
    main()
