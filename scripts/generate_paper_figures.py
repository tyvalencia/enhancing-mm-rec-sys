"""
Generate publication-ready figures from final results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

RESULTS = {
    "LLaVA-NeXT 7B Text-Only": {"recall@10": 0.3539, "ndcg@10": 0.2745},
    "LLaVA+Vision Attention Fusion": {"recall@10": 0.3103, "ndcg@10": 0.2500},
    "LLaVA+Vision (Concat)": {"recall@10": 0.2825, "ndcg@10": 0.1763},
    "LLaVA+Vision (Naive 50/50)": {"recall@10": 0.2806, "ndcg@10": 0.1762},
    "SMORE (LLaVA+Vision)": {"recall@10": 0.2375, "ndcg@10": 0.1652},
    "BERT+Vision Attention Fusion": {"recall@10": 0.2348, "ndcg@10": 0.1966},
    "BERT Text-Only (Baseline)": {"recall@10": 0.2285, "ndcg@10": 0.1838},
    "BERT+Vision (Concat)": {"recall@10": 0.1887, "ndcg@10": 0.1540},
    "SMORE (BERT+Vision)": {"recall@10": 0.1881, "ndcg@10": 0.1398},
    "BERT+Vision (Naive 50/50)": {"recall@10": 0.1803, "ndcg@10": 0.1428},
    "Gating Fusion": {"recall@10": 0.1269, "ndcg@10": 0.0973},
    "Vision-Only": {"recall@10": 0.1053, "ndcg@10": 0.0734},
}

BASELINE_RECALL = 0.2285


# Main performance comparison (Top 8 models)
def create_main_performance_comparison():
    top_models = [
        "LLaVA-NeXT 7B Text-Only",
        "LLaVA+Vision Attention Fusion",
        "LLaVA+Vision (Concat)",
        "LLaVA+Vision (Naive 50/50)",
        "SMORE (LLaVA+Vision)",
        "BERT+Vision Attention Fusion",
        "BERT Text-Only (Baseline)",
        "SMORE (BERT+Vision)",
    ]
    
    models = top_models
    recalls = [RESULTS[m]["recall@10"] for m in models]
    ndcgs = [RESULTS[m]["ndcg@10"] for m in models]
    
    clean_names = []
    for m in models:
        if "LLaVA-NeXT 7B" in m:
            clean_names.append("LLaVA-NeXT\n7B Text-Only")
        elif "LLaVA+Vision Attention" in m:
            clean_names.append("LLaVA+Vision\nAttention Fusion")
        elif "LLaVA+Vision (Concat)" in m:
            clean_names.append("LLaVA+Vision\n(Concat)")
        elif "LLaVA+Vision (Naive" in m:
            clean_names.append("LLaVA+Vision\n(Naive 50/50)")
        elif "SMORE (LLaVA" in m:
            clean_names.append("SMORE\n(LLaVA+Vision)")
        elif "BERT+Vision Attention" in m:
            clean_names.append("BERT+Vision\nAttention Fusion")
        elif "BERT Text-Only" in m:
            clean_names.append("BERT Text-Only\n(Baseline)")
        elif "SMORE (BERT" in m:
            clean_names.append("SMORE\n(BERT+Vision)")
        else:
            clean_names.append(m.replace(" ", "\n"))
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(24, 7.5))
    bars1 = ax.bar(x - width/2, recalls, width, label='Recall@10', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ndcgs, width, label='NDCG@10', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (recall, ndcg) in enumerate(zip(recalls, ndcgs)):
        ax.text(i - width/2, recall + 0.005, f'{recall:.3f}', 
                ha='center', va='bottom', fontsize=5, fontweight='bold')
        ax.text(i + width/2, ndcg + 0.005, f'{ndcg:.3f}', 
                ha='center', va='bottom', fontsize=5, fontweight='bold')
    
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    label_positions = []
    for i in range(len(clean_names)):
        offset = -0.02 if i % 2 == 0 else -0.08
        label_positions.append(offset)
    
    for i, (name, pos) in enumerate(zip(clean_names, label_positions)):
        ax.text(i, pos, name, fontsize=8, ha='center', va='top', rotation=0)
    
    ax.set_xticklabels([''] * len(clean_names))
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, max(max(recalls), max(ndcgs)) * 1.15)
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.18)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_main_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig1_main_performance.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'fig1_main_performance.png'}")
    plt.close()


# Improvement over baseline
def create_improvement_analysis():
    models = [
        "LLaVA-NeXT 7B Text-Only",
        "LLaVA+Vision Attention Fusion",
        "LLaVA+Vision (Concat)",
        "LLaVA+Vision (Naive 50/50)",
        "SMORE (LLaVA+Vision)",
        "BERT+Vision Attention Fusion",
        "SMORE (BERT+Vision)",
        "BERT+Vision (Concat)",
        "BERT+Vision (Naive 50/50)",
        "Gating Fusion",
        "Vision-Only",
    ]
    
    improvements = [((RESULTS[m]["recall@10"] / BASELINE_RECALL) - 1) * 100 for m in models]
    
    clean_names = []
    for m in models:
        if "LLaVA-NeXT 7B" in m:
            clean_names.append("LLaVA-NeXT\n7B Text-Only")
        elif "LLaVA+Vision Attention" in m:
            clean_names.append("LLaVA+Vision\nAttention Fusion")
        elif "LLaVA+Vision (Concat)" in m:
            clean_names.append("LLaVA+Vision\n(Concat)")
        elif "LLaVA+Vision (Naive" in m:
            clean_names.append("LLaVA+Vision\n(Naive)")
        elif "SMORE (LLaVA" in m:
            clean_names.append("SMORE\n(LLaVA)")
        elif "BERT+Vision Attention" in m:
            clean_names.append("BERT+Vision\nAttention")
        elif "SMORE (BERT" in m:
            clean_names.append("SMORE\n(BERT)")
        elif "BERT+Vision (Concat)" in m:
            clean_names.append("BERT+Vision\n(Concat)")
        elif "BERT+Vision (Naive" in m:
            clean_names.append("BERT+Vision\n(Naive)")
        elif "Gating" in m:
            clean_names.append("Gating\nFusion")
        else:
            clean_names.append(m.replace(" ", "\n"))
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(clean_names, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        if width >= 0:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(width - 1, bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%',
                   ha='right', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.set_xlabel('Improvement vs. BERT Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_improvement_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig2_improvement_analysis.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'fig2_improvement_analysis.png'}")
    plt.close()


# Fusion strategy comparison
def create_fusion_strategy_comparison():
    strategies = {
        "Text-Only": ["LLaVA-NeXT 7B Text-Only", "BERT Text-Only (Baseline)"],
        "Naive Fusion": ["LLaVA+Vision (Naive 50/50)", "BERT+Vision (Naive 50/50)"],
        "Concat Fusion": ["LLaVA+Vision (Concat)", "BERT+Vision (Concat)"],
        "Attention Fusion": ["LLaVA+Vision Attention Fusion", "BERT+Vision Attention Fusion"],
        "SMORE": ["SMORE (LLaVA+Vision)", "SMORE (BERT+Vision)"],
    }
    
    llava_recalls = []
    bert_recalls = []
    strategy_names = []
    
    for strategy, models in strategies.items():
        llava_model = [m for m in models if "LLaVA" in m][0]
        bert_model = [m for m in models if "BERT" in m and "LLaVA" not in m][0]
        
        llava_recalls.append(RESULTS[llava_model]["recall@10"])
        bert_recalls.append(RESULTS[bert_model]["recall@10"])
        strategy_names.append(strategy)
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, llava_recalls, width, label='LLaVA-Based', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, bert_recalls, width, label='BERT-Based', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (llava, bert) in enumerate(zip(llava_recalls, bert_recalls)):
        ax.text(i - width/2, llava + 0.01, f'{llava:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width/2, bert + 0.01, f'{bert:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
    ax.set_title('Fusion Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_ylim(0, max(max(llava_recalls), max(bert_recalls)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_fusion_strategies.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig3_fusion_strategies.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'fig3_fusion_strategies.png'}")
    plt.close()


# Ablation study for LLaVA vs BERT
def create_ablation_study():
    comparisons = [
        ("Text-Only", "LLaVA-NeXT 7B Text-Only", "BERT Text-Only (Baseline)"),
        ("Attention Fusion", "LLaVA+Vision Attention Fusion", "BERT+Vision Attention Fusion"),
        ("Concat", "LLaVA+Vision (Concat)", "BERT+Vision (Concat)"),
        ("Naive 50/50", "LLaVA+Vision (Naive 50/50)", "BERT+Vision (Naive 50/50)"),
        ("SMORE", "SMORE (LLaVA+Vision)", "SMORE (BERT+Vision)"),
    ]
    
    strategies = [c[0] for c in comparisons]
    llava_recalls = [RESULTS[c[1]]["recall@10"] for c in comparisons]
    bert_recalls = [RESULTS[c[2]]["recall@10"] for c in comparisons]
    improvements = [((llava - bert) / bert) * 100 for llava, bert in zip(llava_recalls, bert_recalls)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(strategies))
    width = 0.35
    bars1 = ax1.bar(x - width/2, llava_recalls, width, label='LLaVA-Based', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, bert_recalls, width, label='BERT-Based', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (llava, bert) in enumerate(zip(llava_recalls, bert_recalls)):
        ax1.text(i - width/2, llava + 0.01, f'{llava:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax1.text(i + width/2, bert + 0.01, f'{bert:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation: LLaVA vs BERT', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax2.barh(strategies, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax2.text(width + (2 if width >= 0 else -2), bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%',
                ha='left' if width >= 0 else 'right', va='center', fontsize=9, fontweight='bold',
                color='white' if width < 0 else 'black')
    
    ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('LLaVA Advantage', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_ablation_study.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig4_ablation_study.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'fig4_ablation_study.png'}")
    plt.close()


def main():
    print("Figures of metrics:")
    
    print("\n[1/4] Creating main performance comparison")
    create_main_performance_comparison()
    
    print("\n[2/4] Creating improvement analysis")
    create_improvement_analysis()
    
    print("\n[3/4] Creating fusion strategy comparison")
    create_fusion_strategy_comparison()
    
    print("\n[4/4] Creating ablation study")
    create_ablation_study()
    
    print(f"\nFigures saved to: {output_dir}/")
    print("  - fig1_main_performance.png/pdf")
    print("  - fig2_improvement_analysis.png/pdf")
    print("  - fig3_fusion_strategies.png/pdf")
    print("  - fig4_ablation_study.png/pdf")


if __name__ == "__main__":
    main()
