"""
Script to download Amazon product metadata from https://nijianmo.github.io/amazon/index.html
"""

import gzip
import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download Amazon product metadata")
    parser.add_argument("--category", type=str, default="Clothing_Shoes_and_Jewelry",
                       help="Amazon category to download")
    parser.add_argument("--output-dir", type=str, default="data/raw/amazon_metadata",
                       help="Output directory for metadata")
    return parser.parse_args()


def download_metadata(category: str, output_dir: Path):
    # We can add more categories here later for more thorough testing
    category_urls = {
        "Clothing_Shoes_and_Jewelry": "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"
    }
    
    url = category_urls[category]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"meta_{category}.json.gz"
    
    if output_file.exists():
        print(f"✓ Metadata already downloaded: {output_file}")
        return output_file
    
    print(f"Downloading metadata from {url}...")
    
    
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Downloaded to {output_file}")
    return output_file
    

def parse_metadata(metadata_file: Path, output_file: Path):
    print(f"\nParsing metadata from {metadata_file}...")
    
    metadata_dict = {}
    
    with gzip.open(metadata_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing products"):
            try:
                product = json.loads(line.strip())
                asin = product.get('asin')
                
                if asin:
                    # Extract relevant fields for LVLM
                    metadata_dict[asin] = {
                        'title': product.get('title', ''),
                        'description': product.get('description', ''),
                        'features': product.get('feature', []),
                        'price': product.get('price'),
                        'brand': product.get('brand', ''),
                        'categories': product.get('categories', [])
                    }
            except json.JSONDecodeError:
                continue
    
    # Save parsed metadata
    print(f"\nSaving parsed metadata to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"✓ Parsed {len(metadata_dict)} products")
    return metadata_dict


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("AMAZON METADATA DOWNLOADER")
    print("="*60)
    
    metadata_file = download_metadata(args.category, output_dir)
    
    if metadata_file:
        output_file = output_dir / f"meta_{args.category}_parsed.json"
        parse_metadata(metadata_file, output_file)


if __name__ == "__main__":
    main()

