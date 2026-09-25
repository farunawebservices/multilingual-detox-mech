#!/usr/bin/env python3
"""
Stage 19: Create human evaluation samples.
"""
import pandas as pd
import numpy as np
from pathlib import Path

class Config:
    mt5_preds_path = "results/evaluation/mt5_v2_400plus/predictions_seed_2024.csv"
    mt0_preds_path = "results/evaluation/mt0/predictions_seed_2024.csv"
    llama_preds_path = "results/evaluation/llama8b_qlora/seed_2024_predictions.csv"
    output_path = "results/human_evaluation/human_eval_samples.csv"
    n_samples_per_model = 50
    seed = 42

def main():
    config = Config()
    out_dir = Path(config.output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    mt5_df = pd.read_csv(config.mt5_preds_path)
    mt0_df = pd.read_csv(config.mt0_preds_path)
    llama_df = pd.read_csv(config.llama_preds_path)
    
    print(f"Loaded: MT5={len(mt5_df)}, MT0={len(mt0_df)}, Llama={len(llama_df)} examples")
    
    # Sample per language (balanced)
    np.random.seed(config.seed)
    languages = mt5_df['language'].unique()
    n_per_lang = config.n_samples_per_model // len(languages)
    
    samples = []
    
    for lang in languages:
        lang_mt5 = mt5_df[mt5_df['language'] == lang].sample(n=min(n_per_lang, len(mt5_df[mt5_df['language'] == lang])), random_state=config.seed)
        lang_mt0 = mt0_df[mt0_df['language'] == lang].sample(n=min(n_per_lang, len(mt0_df[mt0_df['language'] == lang])), random_state=config.seed)
        lang_llama = llama_df[llama_df['language'] == lang].sample(n=min(n_per_lang, len(llama_df[llama_df['language'] == lang])), random_state=config.seed)
        
        for _, row in lang_mt5.iterrows():
            samples.append({
                'sample_id': f"mt5_{row['pair_id']}",
                'model': 'mT5 v2 (400+)',
                'language': row['language'],
                'toxic_input': row['toxic_input'],
                'reference_detox': row['detox_output'],
                'model_output': row['prediction'],
                'annotator_non_toxic': '',
                'annotator_meaning': '',
                'annotator_fluency': '',
                'annotator_acceptable': '',
                'annotator_comments': ''
            })
        
        for _, row in lang_mt0.iterrows():
            samples.append({
                'sample_id': f"mt0_{row['pair_id']}",
                'model': 'mT0 (400+)',
                'language': row['language'],
                'toxic_input': row['toxic_input'],
                'reference_detox': row['detox_output'],
                'model_output': row['prediction'],
                'annotator_non_toxic': '',
                'annotator_meaning': '',
                'annotator_fluency': '',
                'annotator_acceptable': '',
                'annotator_comments': ''
            })
        
        for _, row in lang_llama.iterrows():
            samples.append({
                'sample_id': f"llama_{row['pair_id']}",
                'model': 'Llama-3-8B QLoRA',
                'language': row['language'],
                'toxic_input': row['toxic_input'],
                'reference_detox': row['detox_output'],
                'model_output': row['prediction'],
                'annotator_non_toxic': '',
                'annotator_meaning': '',
                'annotator_fluency': '',
                'annotator_acceptable': '',
                'annotator_comments': ''
            })
    
    samples_df = pd.DataFrame(samples)
    samples_df.to_csv(config.output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Created: {config.output_path}")
    print(f"  Total samples: {len(samples_df)}")
    print(f"  Per model: {len(samples_df) // 3}")
    print(f"  Per language: {len(samples_df) // len(languages)}")
    print(f"  Languages: {sorted(languages)}")

if __name__ == "__main__":
    main()
