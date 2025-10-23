# Multimodal Recommender System using LVLMs

**Authors:** Ty Valencia, Burak Barlas, Varun Singhal, Ruchir Bhatia

## Description

This pipeline implements a multimodal recommendation system that leverages Large Vision-Language Models (LVLMs) to generate rich product representations. It provides this framework:

1. Processes product data combining titles, images, and user interaction history
2. Generates embeddings using Sentence-BERT for product representations
3. Implements baseline models (title-only and LVLM-enhanced recommenders)
4. Evaluates recommendation quality using standard metrics (Recall@K, NDCG@K, Hit Rate@K)
5. Supports integration with commercial LVLMs (Gemini, GPT-4V) or open-source models (LLaVA)

---

## Repository Structure
```text
config/
  default.yaml           # experiment configuration and model parameters
data/
  raw/                   # raw input data (interactions.csv)
  processed/             # train/test splits
  summaries/             # LVLM-generated product descriptions
scripts/
  prepare_data.py        # data preprocessing and train/test splits
  run_experiment.py      # main experiment runner, code orchestrator
  generate_summaries.py  # LVLM summary generation pipeline
  test_setup.py          # verify installation and dependencies
src/
  __init__.py
  dataset.py             # data loading and processing utilities
  recommender.py         # baseline recommendation models
  metrics.py             # evaluation metrics (Recall, NDCG, Hit Rate)
results/                 # experiment outputs and performance logs
requirements.txt         # dependencies to install
README.md                # this documentation
LICENSE                  # MIT license
```

---

## How to Run

### Install dependencies

`pip install -r requirements.txt` <br>

### Configure environment (optional)

For LVLMs, create a `.env` file: <br>
`GOOGLE_API_KEY=your_api_key_here` (for Gemini) <br>
`OPENAI_API_KEY=your_api_key_here` (for GPT-4V) <br>

### Prepare data

`python scripts/prepare_data.py` <br>

### Run experiments

`python scripts/run_experiment.py` 

Which comes with the following flags: <br>

#### --model-type
Specifies which recommendation model to use. <br>
`title_only` runs the baseline text-only model <br>
`lvlm_enhanced` runs the multimodal model with LVLM summaries <br>

#### --device
Specifies compute device. <br>
`cuda` uses GPU acceleration <br>
`cpu` uses CPU only <br>

#### --config
Path to custom YAML configuration file (default: `config/default.yaml`).

---

## Baseline Evaluation Scores

**Title-Only Baseline:**
* Recall@5: TBD
* Recall@10: TBD
* Recall@20: TBD
* NDCG@5: TBD
* NDCG@10: TBD
* NDCG@20: TBD
* Hit Rate@5: TBD
* Hit Rate@10: TBD
* Hit Rate@20: TBD

**LVLM-Enhanced Model:**
* Recall@5: TBD
* Recall@10: TBD
* Recall@20: TBD
* NDCG@5: TBD
* NDCG@10: TBD
* NDCG@20: TBD
* Hit Rate@5: TBD
* Hit Rate@10: TBD
* Hit Rate@20: TBD

---

## Getting Data

**📖 See [DATA_GUIDE.md](DATA_GUIDE.md) for detailed setup instructions with code examples.**

### Recommended Datasets

**1. Amazon Product Dataset (2018)** <br>
Multi-category products with images, reviews, and interaction histories. <br>
Download: [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html) <br>
Choose a category (e.g., "Electronics", "Clothing"), download the review data and metadata.

**2. Amazon Reviews 2023** <br>
Latest version with more products and reviews. <br>
Download: [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)

**3. Kaggle Datasets** <br>
Search for "product recommendation" or "e-commerce": <br>
[https://www.kaggle.com/datasets?search=product+recommendation](https://www.kaggle.com/datasets?search=product+recommendation)

### Data Format

Your CSV should be named `interactions.csv` and placed in `data/raw/` with these columns:

```csv
user_id,item_id,title,image_path,timestamp
user_1,item_42,Blue Cotton T-Shirt,images/item_42.jpg,1609459200
```

**Required columns:**
- `user_id`: User identifier
- `item_id`: Item identifier  
- `title`: Product title

**Optional columns:**
- `image_path`: Path to product image
- `timestamp`: Interaction timestamp (UNIX timestamp or ISO format)

---

## References

[Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models](https://arxiv.org/abs/2402.15809)

[SMORE: Spectrum-based Modality Representation Fusion](https://arxiv.org/abs/2412.15442)

[MMREC: LLM Based Multi-Modal Recommender System](https://arxiv.org/abs/2408.04211)

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
