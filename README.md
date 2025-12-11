# Enhancing Multimodal Recommendation Systems using Large Vision-Language Models

**Authors:** Burak Barlas, Varun Singhal, Ty Valencia, Ruchir Bhatia  
**Institution:** University of Southern California  
**Course:** CS 566: Deep Learning and Its Applications

## Abstract

We investigate how large vision-language models (LVLMs) can enhance multimodal recommender systems by generating rich textual descriptions from product images, which are then encoded into semantic embeddings for efficient retrieval. Building on recent work in LVLM-based recommendation (e.g., Rec-GPT4V) and spectral multimodal fusion (SMORE), we propose using LLaVA-NeXT 7B to extract visual semantics from product images in text form, which we encode using Sentence-BERT for downstream recommendation.

We compare this approach against traditional text-only baselines (BERT title embeddings), various vision-text fusion strategies (naive averaging, attention-based fusion, spectral fusion via SMORE), and vision-only baselines within a unified lightweight retrieval framework.

On a subset of 4,708 items (12.2% of the catalog) from the Kaggle Multimodal Recommendation dataset for *Clothing, Shoes, and Jewelry*, we evaluate 2,914 users and find that LVLM-generated text descriptions consistently outperform all baselines and fusion approaches. Our best model achieves **0.3539 Recall@10**, representing a **54.9% improvement** over the BERT text-only baseline (0.2285), while also outperforming sophisticated fusion methods including real SMORE implementation and learned attention fusion.

Critically, we find that text-only LVLM descriptions surpass all multimodal fusion strategies, suggesting that LVLMs effectively encode visual information in text form, making additional visual embeddings redundant or even detrimental.

## Key Contributions

1. **Practical LVLM Pipeline**: We propose using LLaVA-NeXT 7B to generate rich textual descriptions from product images offline, which are then encoded with Sentence-BERT for efficient recommendation. This approach bridges the gap between expensive online LVLM inference and simple pre-computed embeddings.

2. **Comprehensive Evaluation**: We conduct a comprehensive evaluation comparing 12 different item representation strategies, including text-only baselines, various vision-text fusion methods (naive, attention-based, spectral), and vision-only approaches, all within a unified retrieval framework.

3. **Strong Empirical Results**: Through experiments on 4,708 items and 2,914 users, we demonstrate that LVLM-generated text descriptions achieve 0.3539 Recall@10, a 54.9% improvement over BERT baseline (0.2285), while outperforming all fusion strategies.

4. **Insightful Finding**: We provide evidence that text-based LVLM descriptions capture visual semantics more effectively than traditional vision embeddings, making additional multimodal fusion unnecessary or even counterproductive in our setting.

## Repository Structure

```
enhancing-mm-rec-sys/
├── config/
│   └── default.yaml              # Experiment configuration
├── data/
│   ├── raw/                      # Raw datasets (download separately)
│   ├── processed/                # Preprocessed data (generated)
│   └── summaries/                # LLaVA-generated summaries (generated)
├── figures/                      # Paper figures
│   ├── fig1_main_performance.png
│   ├── fig2_improvement_analysis.png
│   ├── fig3_fusion_strategies.png
│   └── fig4_ablation_study.png
├── scripts/
│   ├── prepare_data.py           # Preprocess Kaggle data
│   ├── download_amazon_metadata.py  # Download Amazon metadata
│   ├── match_metadata.py         # Match items with metadata
│   ├── generate_llava_next_summaries.py  # Generate LLaVA descriptions
│   ├── encode_llava_next_summaries.py    # Encode with Sentence-BERT
│   ├── train_fusion_bert_vision.py       # Train fusion models
│   ├── evaluate_all_final.py     # Comprehensive evaluation
│   ├── run_experiment.py         # Main experiment runner
│   └── generate_paper_figures.py  # Generate paper figures
├── src/
│   ├── dataset.py                # Data loading utilities
│   ├── metrics.py                # Evaluation metrics
│   ├── recommender.py            # Recommender models
│   └── models/
│       ├── attention_fusion.py   # Attention-based fusion
│       ├── smore.py              # SMORE implementation
│       └── base_model.py         # Base model classes
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for LLaVA generation)
- 16GB+ VRAM for LLaVA-NeXT 7B with 4-bit quantization

### Setup

```bash
# Clone repository
git clone <repository-url>
cd enhancing-mm-rec-sys

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. Download Kaggle Dataset

Download the [Kaggle Multimodal Recommendation System Dataset](https://www.kaggle.com/competitions/multimodal-recommendation) and extract to `data/raw/archive/`.

### 2. Prepare Data

```bash
# Preprocess Kaggle data into train/val/test splits
python scripts/prepare_data.py
```

### 3. Download Amazon Metadata (Optional)

For generating richer summaries:

```bash
# Download Amazon product metadata
python scripts/download_amazon_metadata.py --category Clothing_Shoes_and_Jewelry

# Match items with metadata
python scripts/match_metadata.py
```

## Generating LLaVA-NeXT Descriptions

### Step 1: Generate Text Descriptions

```bash
# Generate descriptions using LLaVA-NeXT 7B (4-bit quantization)
python scripts/generate_llava_next_summaries.py --model 7b --batch-size 4

# Options:
#   --model: "7b" or "13b" (default: "7b")
#   --batch-size: Batch size for generation (default: 4 for 7B)
#   --max-items: Maximum items to process (default: all)
```

**Note:** Generation takes approximately 8 hours for 4,708 items on RTX 5080 with 4-bit quantization.

### Step 2: Encode with Sentence-BERT

```bash
# Encode LLaVA summaries into 384d embeddings
python scripts/encode_llava_next_summaries.py --model 7b
```

## Training Fusion Models

```bash
# Train attention-based fusion models
python scripts/train_fusion_bert_vision.py
```

This trains:
- BERT+Vision Attention Fusion
- LLaVA+Vision Attention Fusion
- Gating Fusion
- Cross-Modal Attention

## Running Experiments

### Basic Usage

```bash
# Run single model evaluation
python scripts/run_experiment.py --model-type title_only --device cpu
python scripts/run_experiment.py --model-type lvlm_enhanced --device cpu
python scripts/run_experiment.py --model-type smore_fusion --device cpu
```

### Comprehensive Evaluation

```bash
# Evaluate all 12 models
python scripts/evaluate_all_final.py
```

This evaluates:
1. LLaVA-NeXT 7B Text-Only
2. BERT Text-Only (Baseline)
3. LLaVA+Vision Attention Fusion
4. LLaVA+Vision (Concat)
5. LLaVA+Vision (Naive 50/50)
6. SMORE (LLaVA+Vision)
7. BERT+Vision Attention Fusion
8. BERT+Vision (Concat)
9. BERT+Vision (Naive 50/50)
10. SMORE (BERT+Vision)
11. Gating Fusion
12. Vision-Only

## Experimental Results

### Dataset

- **Category**: Clothing, Shoes, and Jewelry
- **Full Dataset**: 23,318 users, 38,493 items
- **LLaVA Coverage**: 4,708 items (12.2% of catalog)
- **Evaluation Users**: 2,914 users (with LLaVA items in both train and test)

### Performance Comparison

| Rank | Model | Recall@10 | NDCG@10 | vs BERT |
|------|-------|----------|---------|---------|
| **1** | **LLaVA-NeXT 7B Text-Only** | **0.3539** | **0.2745** | **+54.9%** |
| 2 | LLaVA+Vision Attention Fusion | 0.3103 | 0.2500 | +35.8% |
| 3 | LLaVA+Vision (Concat) | 0.2825 | 0.1763 | +23.6% |
| 4 | LLaVA+Vision (Naive 50/50) | 0.2806 | 0.1762 | +22.8% |
| 5 | SMORE (LLaVA+Vision) | 0.2375 | 0.1652 | +3.9% |
| 6 | BERT+Vision Attention Fusion | 0.2348 | 0.1966 | +2.8% |
| **7** | **BERT Text-Only (Baseline)** | **0.2285** | **0.1838** | **---** |
| 8 | BERT+Vision (Concat) | 0.1887 | 0.1540 | -17.4% |
| 9 | SMORE (BERT+Vision) | 0.1881 | 0.1398 | -17.7% |
| 10 | BERT+Vision (Naive 50/50) | 0.1803 | 0.1428 | -21.1% |
| 11 | Gating Fusion | 0.1269 | 0.0973 | -44.5% |
| 12 | Vision-Only | 0.1053 | 0.0734 | -53.9% |

### Key Findings

1. **LLaVA-NeXT Text-Only Dominates**: Achieves 0.3539 Recall@10, a 54.9% improvement over BERT baseline.

2. **Text-Only Beats All Fusion**: Remarkably, text-only LLaVA outperforms all fusion approaches, including sophisticated methods like attention fusion and real SMORE implementation.

3. **Fusion Can Hurt Performance**: Naive 50/50 fusion of BERT text with vision embeddings actually degrades performance compared to BERT alone.

4. **Learned Fusion Helps But Not Enough**: Attention-based fusion improves over naive fusion, but still falls short of text-only LLaVA.

5. **LLaVA Benefits from Better Text Representations**: Even when fusion strategies are applied, LLaVA-based variants consistently outperform their BERT-based counterparts.

## Generating Paper Figures

```bash
# Generate all paper figures
python scripts/generate_paper_figures.py
```

This creates:
- `figures/fig1_main_performance.png` - Main performance comparison
- `figures/fig2_improvement_analysis.png` - Improvement analysis
- `figures/fig3_fusion_strategies.png` - Fusion strategy comparison
- `figures/fig4_ablation_study.png` - Ablation study

## Implementation Details

### LLaVA-NeXT Generation
- **Model**: `lmms-lab/llava-next-7b-hf`
- **Quantization**: 4-bit (BitsAndBytes)
- **Batch Size**: 4
- **Hardware**: NVIDIA RTX 5080 (16GB VRAM)
- **Generation Time**: ~8 hours for 4,708 items
- **Prompt**: "Describe this product image in detail, focusing on visual attributes like color, style, material, and what occasion it might be suitable for."

### Sentence-BERT Encoding
- **Model**: `all-MiniLM-L6-v2`
- **Pooling**: Mean pooling
- **Batch Size**: 128
- **Output Dimension**: 384

### Attention Fusion Training
- **Epochs**: 15
- **Batch Size**: 256
- **Loss**: Contrastive loss
- **Optimizer**: Adam

### SMORE Implementation
- **Graph Construction**: k-nearest neighbors (k=10)
- **GCN Layers**: 2
- **Spectral Filtering**: Frequency domain projection
- **Training Epochs**: 20

## Limitations

1. **Coverage Constraint**: Only 4,708 items (12.2% of catalog) have LLaVA descriptions due to computational constraints.

2. **Computational Cost**: Generating LLaVA descriptions requires approximately 8 hours for 4,708 items on RTX 5080. Scaling to full catalog (38,493 items) would require roughly 70 hours.

3. **Evaluation Subset**: Fair comparison required evaluating on 2,914 users (those with LLaVA items in both training and test sets).

4. **Model Choice**: We used LLaVA-NeXT 7B with 4-bit quantization. Larger models (e.g., 13B) or higher precision might improve results further.

## Future Work

- Generate LLaVA descriptions for complete item catalog (38,493 items)
- Evaluate LLaVA-NeXT 13B or other state-of-the-art LVLMs
- User and item slice analysis (activity levels, popularity, categories)
- Statistical significance testing
- Prompt engineering for improved descriptions
- Hybrid approaches combining LLaVA text with other signals
- Sequential user modeling with learned sequence models

## References

- [Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models](https://arxiv.org/abs/2402.15809)
- [SMORE: Spectrum-based Modality Representation Fusion](https://arxiv.org/abs/2412.15442)
- [MMREC: LLM Based Multi-Modal Recommender System](https://arxiv.org/abs/2408.04211)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{barlas2024enhancing,
  title={Enhancing Multimodal Recommendation Systems using Large Vision-Language Models},
  author={Barlas, Burak and Singhal, Varun and Valencia, Ty and Bhatia, Ruchir},
  journal={CS 566: Deep Learning and Its Applications --- Final Project Report},
  year={2025},
  institution={University of Southern California}
}
```

## License

MIT License - see LICENSE file for details.
