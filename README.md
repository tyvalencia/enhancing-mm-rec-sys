# Multimodal Recommender System using LVLMs

**Authors:** Burak Barlas, Varun Singhal, Ty Valencia, Ruchir Bhatia

## Description

This pipeline implements a multimodal recommendation system that uses rich product metadata (title, brand, features, description) to create semantic representations for improved recommendations. The repository contains this framework:

1. Processes product data combining text, images, and user interaction history
2. Generates embeddings using Sentence-BERT for semantic product representations
3. Implements three baseline models: title-only, SMORE fusion, and LVLM-enhanced
4. Evaluates recommendation quality using standard metrics (Recall@K, NDCG@K, Hit Rate@K)
5. Supports integration with commercial LVLMs (Gemini, GPT-4V) or open-source models (LLaVA)

---

## Repository Structure
```text
config/
  default.yaml                   # experiment configuration and model parameters
data/
  raw/
    archive/                     # Kaggle multimodal datasets (download separately)
    amazon_metadata/             # Amazon product metadata (download via the download_amazon_metadata script)
  processed/                     # preprocessed train/test splits and embeddings
  summaries/                     # product summaries (metadata or LVLM-generated)
scripts/
  prepare_data.py                # preprocess Kaggle data, create splits
  download_amazon_metadata.py    # download Amazon product metadata
  match_metadata.py              # match Kaggle items with Amazon metadata
  create_metadata_summaries.py   # create rich non-LLM summaries from metadata
  generate_summaries.py          # generate summaries with LLMs (Gemini/GPT-4V) (optional atm, optimize)
  run_experiment.py              # main orchestrator to run experiments
src/
  __init__.py
  dataset.py                     # data loading utilities
  recommender.py                 # baseline models (Title, SMORE, LVLM)
  metrics.py                     # evaluation metrics (Recall, NDCG, Hit Rate)
results/                         # experiment outputs
requirements.txt                 # Python dependencies
README.md                        # this documentation
LICENSE                          # MIT license
```

---

## How to Run

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare Kaggle dataset
python scripts/prepare_data.py

# 3. Download Amazon metadata and create summaries
python scripts/download_amazon_metadata.py --category Clothing_Shoes_and_Jewelry # good baseline, can also download other data
python scripts/match_metadata.py
python scripts/create_metadata_summaries.py

# 4. Run experiments
python scripts/run_experiment.py --model-type title_only --device cpu
python scripts/run_experiment.py --model-type smore_fusion --device cpu
python scripts/run_experiment.py --model-type lvlm_enhanced --device cpu

# 4a. Optional flags:
- `--model-type`: Model to evaluate (`title_only` | `smore_fusion` | `lvlm_enhanced`)
- `--device`: Compute device (`cpu` | `cuda`)
- `--use-validation`: Evaluate on validation set instead of test

# Results saved to results/*.txt
```

### Optional: LLM-based Summaries

For research comparing metadata vs. LLM-generated summaries:

```bash
# Setup .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Generate LLM summaries (requires API key, too slow, better implementation with internal models to come)
python scripts/generate_summaries.py --model gemini --max-items 100
```

---

## Data Sources

### 1. Kaggle Multimodal Dataset (Required)

**Download:** [Kaggle Multimodal Recommendation System Dataset](https://www.kaggle.com/datasets/ignacioavas/alignmacrid-vae) (6GB)

### 2. Amazon Product Metadata (For LVLM)

**Source:** [Amazon Review Dataset (2018)](https://nijianmo.github.io/amazon/index.html)

Provides real product titles, descriptions, and features for generating meaningful LVLM summaries instead of generic text.

---

## Experimental Results

**Dataset:** Clothing, Shoes & Jewelry (23,318 users, 38,493 items)

### Performance Comparison

| Model | Recall@5 | NDCG@5 | Hit Rate@5 | Recall@10 | NDCG@10 | Hit Rate@10 |
|-------|----------|--------|------------|-----------|---------|-------------|
| **lvlm_enhanced** | **0.1962** | **0.1612** | **0.2380** | **0.2218** | **0.1700** | **0.2705** |
| title_only | 0.1805 | 0.1479 | 0.2206 | 0.2090 | 0.1575 | 0.2539 |
| smore_fusion | 0.1412 | 0.1139 | 0.1747 | 0.1720 | 0.1243 | 0.2118 |

---

## References

[Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models](https://arxiv.org/abs/2402.15809)

[SMORE: Spectrum-based Modality Representation Fusion](https://arxiv.org/abs/2412.15442)

[MMREC: LLM Based Multi-Modal Recommender System](https://arxiv.org/abs/2408.04211)

[Amazon Review Dataset (2018)](https://nijianmo.github.io/amazon/index.html)

---

## Citation

```
@article{barlas2024enhancing,
  title={Enhancing Multimodal Recommendation Systems using Large Vision-Language Models},
  author={Barlas, Burak and Singhal, Varun and Valencia, Ty and Bhatia, Ruchir},
  journal={CSCI 566 Course Project},
  year={2024}
}
```
