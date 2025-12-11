"""
Encode LLaVA-NeXT summaries using Sentence-BERT.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Encode LLaVA-NeXT summaries")
    parser.add_argument("--model", type=str, default="7b", choices=["7b", "13b"], help="Model size used for summaries (default: 7b)")
    args = parser.parse_args()

    print(f"Encoding LLaVA-NeXT-{args.model.upper()} Summaries")
    
    # Load summaries
    print("\n1. Loading LLaVA-NeXT summaries")
    summary_file = f'data/summaries/llava_next_{args.model}_summaries.json'
    with open(summary_file, 'r') as f:
        summaries = json.load(f)
    print(f"{len(summaries)} summaries found")
    
    # Load model
    print("\n2. Loading Sentence-BERT model")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded, embedding dimensions: {model.get_sentence_embedding_dimension()}")
    
    # Extract texts and IDs
    item_ids = []
    texts = []
    for item_id, data in summaries.items():
        item_ids.append(int(item_id))
        combined = f"{data['title']}. {data['summary']}" # Combine title and summary for richer representation
        texts.append(combined)
    
    # Generate embeddings
    print(f"\n3. Generating embeddings for {len(texts)} items")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create mapping
    mapping = {'item_ids': item_ids}
    
    # Save
    print("\n4. Saving results")
    embedding_file = f'data/processed/llava_next_{args.model}_text_embeds.npy'
    mapping_file = f'data/processed/llava_next_{args.model}_mapping.json'
    
    np.save(embedding_file, embeddings)
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Finished encoding LLaVA-NeXT-{args.model.upper()}. Saved to:")
    print(f" - Embeddings: {embedding_file}")
    print(f" - Mapping: {mapping_file}")
    print(f" - Shape: {embeddings.shape}")

if __name__ == '__main__':
    main()