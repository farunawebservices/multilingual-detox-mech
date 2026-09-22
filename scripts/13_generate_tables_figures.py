#!/usr/bin/env python3
"""
Stage 13: Generate publication-ready tables and figures.
Table 1: Dataset statistics
Table 2: Main automatic evaluation
Figure 1: Copy rate by model/language
Figure 2: 175 vs 400+ comparison
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    corpus_path = "data/frozen_v2_9lang_irregular_yo_xh/multilingual_detox_9lang_irregular_yo_xh.csv"
    eval_mt5_path = "results/evaluation/mt5_v2_400plus/per_example_metrics.csv"
    eval_mt0_path = "results/evaluation/mt0/per_example_metrics.csv"
    eval_175_path = "results/evaluation/mt5/per_example_metrics.csv"  # Historical 175 pairs
    output_dir = "results/tables_figures"

def load_corpus(config):
    df = pd.read_csv(config.corpus_path, keep_default_na=False)
    return df

def table1_dataset_stats(config):
    """Table 1: Dataset and split statistics per language."""
    df = load_corpus(config)
    
    stats = []
    for lang, g in df.groupby("language"):
        unique_inputs = g["toxic_input"].nunique()
        total_pairs = len(g)
        
        stats.append({
            "language": lang,
            "source": "MultiParaDetox" if lang in ["am", "ar", "de", "en", "es", "hi", "uk"] else "Human-annotated",
            "total_pairs": total_pairs,
            "unique_toxic_inputs": unique_inputs,
            "repeated_inputs": total_pairs - unique_inputs,
        })
    
    table1 = pd.DataFrame(stats)
    return table1

def table2_automatic_eval(config):
    """Table 2: Main automatic evaluation per model and language."""
    # Load all evaluation results
    mt5 = pd.read_csv(config.eval_mt5_path)
    mt0 = pd.read_csv(config.eval_mt0_path)
    mt5_175 = pd.read_csv(config.eval_175_path)
    
    # Aggregate by model and language
    def aggregate(df, model_name):
        df["model"] = model_name
        return df
    
    mt5_agg = mt5.groupby(["model", "language"]).agg({
        "sta": "mean", "sim": "mean", "fl": "mean", "j": "mean", "is_copy": "mean"
    }).round(3).reset_index()
    
    mt0_agg = mt0.groupby(["model", "language"]).agg({
        "sta": "mean", "sim": "mean", "fl": "mean", "j": "mean", "is_copy": "mean"
    }).round(3).reset_index()
    
    # Combine
    table2 = pd.concat([mt5_agg, mt0_agg], ignore_index=True)
    return table2

def figure1_copy_rate(config):
    """Figure 1: Copy rate by model and language."""
    mt5 = pd.read_csv(config.eval_mt5_path)
    mt0 = pd.read_csv(config.eval_mt0_path)
    
    mt5["model"] = "mT5-400+"
    mt0["model"] = "mT0-400+"
    
    all_data = pd.concat([mt5, mt0], ignore_index=True)
    
    # Aggregate by model and language
    agg = all_data.groupby(["model", "language"])["is_copy"].mean().unstack().T
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    agg.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Copy Rate")
    ax.set_xlabel("Language")
    ax.set_title("Copy Rate by Model and Language (400+ pairs)")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    
    fig_path = Path(config.output_dir) / "fig1_copy_rate.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")
    return fig_path

def figure2_175_vs_400(config):
    """Figure 2: Historical 175 vs 400+ performance change."""
    mt5_175 = pd.read_csv(config.eval_175_path)
    mt5_400 = pd.read_csv(config.eval_mt5_path)
    
    # Aggregate by language
    agg_175 = mt5_175.groupby("language").agg({"is_copy": "mean", "j": "mean"}).reset_index()
    agg_400 = mt5_400.groupby("language").agg({"is_copy": "mean", "j": "mean"}).reset_index()
    
    agg_175.columns = ["language", "copy_175", "j_175"]
    agg_400.columns = ["language", "copy_400", "j_400"]
    
    comparison = agg_175.merge(agg_400, on="language")
    comparison["copy_change"] = comparison["copy_400"] - comparison["copy_175"]
    comparison["j_change"] = comparison["j_400"] - comparison["j_175"]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Copy rate change
    ax1.bar(comparison["language"], comparison["copy_change"], color="red", alpha=0.7)
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Copy Rate Change (400+ minus 175)")
    ax1.set_xlabel("Language")
    ax1.set_title("Copy Rate: 175 vs 400+ pairs")
    ax1.tick_params(axis="x", rotation=45)
    
    # J score change
    ax2.bar(comparison["language"], comparison["j_change"], color="green", alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("J Score Change (400+ minus 175)")
    ax2.set_xlabel("Language")
    ax2.set_title("J Score: 175 vs 400+ pairs")
    ax2.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    
    fig_path = Path(config.output_dir) / "fig2_175_vs_400.png"
    fig.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")
    return fig_path, comparison

def main():
    config = Config()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Table 1: Dataset statistics...")
    table1 = table1_dataset_stats(config)
    table1.to_csv(out_dir / "table1_dataset_stats.csv", index=False)
    print(table1.to_string(index=False))
    
    print("\nGenerating Table 2: Automatic evaluation...")
    table2 = table2_automatic_eval(config)
    table2.to_csv(out_dir / "table2_automatic_eval.csv", index=False)
    print(table2.to_string(index=False))
    
    print("\nGenerating Figure 1: Copy rate...")
    fig1_path = figure1_copy_rate(config)
    
    print("\nGenerating Figure 2: 175 vs 400+ comparison...")
    fig2_path, comparison = figure2_175_vs_400(config)
    print("\n175 vs 400+ changes:")
    print(comparison.to_string(index=False))
    
    print(f"\n✓ All tables and figures saved to {out_dir}")

if __name__ == "__main__":
    main()
