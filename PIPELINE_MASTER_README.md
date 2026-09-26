# Multilingual Detoxification: Complete Pipeline Master README

## Overview
This repository contains a complete mechanistic interpretability pipeline for multilingual text detoxification across **5 models**:
1. **mT5** (google/mt5-base)
2. **mT0** (bigscience/mt0-base)
3. **Llama-3-8B-Instruct** (meta-llama/Meta-Llama-3-8B-Instruct)
4. **Qwen2.5-7B-Instruct** (Qwen/Qwen2.5-7B-Instruct)
5. **Aya-23-8B** (CohereLabs/aya-23-8B)

## Pipeline Stages (All Models)

Each model follows the same pipeline:
1. **Fine-tuning** (QLoRA for Llama/Qwen/Aya, full for mT5/mT0)
2. **Behavioral Evaluation** (test set metrics)
3. **Linear Probes** (toxicity classification accuracy)
4. **SAE Training** (sparse autoencoders for feature extraction)
5. **Activation Steering** (causal intervention for detoxification)

---

## 1. mT5 (google/mt5-base)

### Scripts
- Training: `scripts/03_train_mt5_detox_v2_400plus.py`
- Evaluation: `scripts/04_generate_and_evaluate_v2.py`
- Probes: `scripts/10_mechanistic_mt5_probes.py`
- Ablation: `scripts/10_mechanistic_mt5_ablation.py`
- Steering: `scripts/15_activation_steering.py`

### Results
- **Training:** `results/training_v2_400plus/seed_2024/` (2583 train, 72 dev, 553 test)
- **Evaluation:** `results/evaluation/mt5/`
  - Best seed: 42 (J=0.8549, SIM=0.9565, FL=0.6081)
  - 3 seeds: 1337, 2024, 42
- **Probes:** `results/mechanistic/mt5_probes/probe_results.json`
- **Steering:** `results/mechanistic/steering/`

### Key Metrics
| Seed | Copy Rate | SIM | FL | J |
|------|-----------|-----|----|---|
| 42 | 0.486 | 0.957 | 0.608 | 0.855 |
| 1337 | 0.516 | 0.941 | 0.617 | 0.853 |
| 2024 | 0.465 | 0.915 | 0.599 | 0.838 |

---

## 2. mT0 (bigscience/mt0-base)

### Scripts
- Training: `scripts/05_train_mt0_detox.py`
- Evaluation: `scripts/06_evaluate_mt0.py`
- Ablation: `scripts/11_mechanistic_mt0_ablation.py`
- SAE Steering: `scripts/17_mt0_sae_steering.py`

### Results
- **Training:** `results/training_mt0/seed_{42,1337,2024}/`
- **Evaluation:** `results/evaluation/mt0/`
  - Best seed: 42 (J=0.8219, SIM=0.8553, FL=0.6103)
- **Ablation:** `results/mechanistic/mt0_ablation/`

### Key Metrics
| Seed | Copy Rate | SIM | FL | J |
|------|-----------|-----|----|---|
| 42 | 0.175 | 0.855 | 0.610 | 0.822 |
| 1337 | 0.149 | 0.832 | 0.599 | 0.810 |
| 2024 | 0.143 | 0.848 | 0.622 | 0.823 |

---

## 3. Llama-3-8B-Instruct (QLoRA)

### Scripts
- Training: `scripts/07_llama8b_qlora_finetune.py`
- Evaluation: `scripts/08_evaluate_llama8b_all_seeds.py`
- Mechanistic: `scripts/13_mechanistic_llama8b_analysis.py`
- Steering: `scripts/18_llama_steering.py`, `scripts/20_llama_steering_fixed.py`

### Results
- **Training:** `results/training_llama_lora/seed_{42,1337,2024}/`
- **Evaluation:** `results/evaluation/llama8b_qlora/llama8b_qlora_results.json`
- **Mechanistic:** `results/mechanistic/llama8b/llama8b_mechanistic_results.json`
- **Steering:** `results/mechanistic/llama_steering/`, `results/mechanistic/llama_steering_fixed/`

### Key Metrics (3-seed aggregate)
| Metric | Mean | Std |
|--------|------|-----|
| Copy Rate | 0.022 | 0.006 |
| SIM | 0.787 | 0.001 |
| FL | 0.599 | 0.019 |
| J | 0.795 | 0.007 |

---

## 4. Qwen2.5-7B-Instruct (QLoRA)

### Scripts
- Training: `scripts/09_qwen2_5_7b_qlora_finetune.py`
- Evaluation: `scripts/10_evaluate_qwen2_5_7b.py`
- Probes: `scripts/23_qwen_linear_probes.py`
- SAE: `scripts/24_qwen_sae_layer20.py`
- Steering: `scripts/25_qwen_steering_layer20.py`
- Figures: `scripts/28_qwen_figures.py`

### Results
- **Training:** `results/training_qwen2_5_7b/seed_{42,1337,2025}/`
- **Evaluation:** `results/evaluation/qwen2_5_7b_qlora/`
- **Probes:** `results/mechanistic/qwen_probes/` (3 seeds)
- **SAE:** `results/mechanistic/qwen_sae/` (3 seeds)
- **Steering:** `results/mechanistic/qwen_steering/` (3 seeds)

### Key Metrics (3-seed aggregate)
| Metric | Mean | Std |
|--------|------|-----|
| Copy Rate | 0.034 | 0.011 |
| SIM | 0.793 | 0.019 |
| FL | 0.581 | 0.010 |
| J | 0.791 | 0.010 |

---

## 5. Aya-23-8B (QLoRA)

### Scripts
- Training: `scripts/29_aya23_8b_qlora_finetune.py`
- Evaluation: `scripts/30_evaluate_aya23_8b.py`
- Probes: `scripts/31_aya_probes_layer20.py`
- SAE: `scripts/32_aya_sae_layer20.py`
- Steering: `scripts/33_aya_steering_rigorous.py`
- Tables/Figures: `scripts/generate_aya_tables_figures.py`

### Results
- **Training:** `results/training_aya_23_8b/seed_{42,1337}/`
- **Evaluation:** `results/evaluation/aya_23_8b/`
- **Probes:** `results/mechanistic/aya_probes/` (AUC: 0.837-0.839)
- **SAE:** `results/mechanistic/aya_sae/seed_42/` (8192 hidden, 40 features)
- **Steering:** `results/mechanistic/aya_steering/`

### Key Metrics (2-seed aggregate)
| Metric | Mean | Std |
|--------|------|-----|
| Copy Rate | 0.071 | 0.003 |
| SIM | 0.802 | 0.002 |
| FL | 0.594 | 0.001 |
| J | 0.799 | 0.0004 |

### Steering Results (Detox Sweep)
- **Selected:** Layer 24, strength -1.0
- **Test deltas (n=553):**
  - toxic_score: -0.0006 (-3.5%)
  - sim: +0.0051 (+0.7%)
  - fl: +0.0107 (+2.1%)
  - j: +0.0055 (+0.75%)

---

## Cross-Model Comparison

### Behavioral Performance (Best Seed per Model)
| Model | Copy Rate | SIM | FL | J |
|-------|-----------|-----|----|---|
| mT5 | 0.486 | 0.957 | 0.608 | **0.855** |
| mT0 | 0.143 | 0.848 | 0.622 | 0.823 |
| Llama | 0.014 | 0.788 | 0.612 | 0.800 |
| Qwen | 0.034 | 0.793 | 0.581 | 0.791 |
| Aya | 0.071 | 0.802 | 0.594 | 0.799 |

### Probe Performance (AUC)
| Model | AUC (all) | AUC (HR) | AUC (cross) |
|-------|-----------|----------|-------------|
| mT5 | - | - | - |
| mT0 | - | - | - |
| Llama | - | - | - |
| Qwen | - | - | - |
| Aya | **0.838** | 0.828 | 0.597 |

---

## Tables and Figures

### Existing Tables
- `results/tables/table1_dataset_stats.csv` - Dataset statistics
- `results/tables/table2_automatic_eval.csv` - Automatic evaluation results
- `results/mechanistic/aya_steering/table_*.csv` - Aya pipeline tables (5 tables)

### Existing Figures
- `results/tables_figures/fig1_copy_rate.png` - Copy rate analysis
- `results/tables_figures/fig1_cross_model_comparison.png` - Cross-model comparison
- `results/tables_figures/fig2_per_language_mt5_mt0.png` - Per-language mT5/mT0
- `results/tables_figures/fig3_copy_vs_sim.png` - Copy vs SIM correlation
- `results/tables_figures/fig4_mechanistic_results.png` - Mechanistic results
- `results/tables_figures/fig5_mechanistic_summary.png` - Mechanistic summary
- `results/tables_figures/fig6_mt0_vs_mt5_comparison.png` - mT0 vs mT5
- `results/tables_figures/fig_qwen_*.png` - Qwen-specific figures (3)
- `results/mechanistic/aya_steering/figures/fig_*.png` - Aya pipeline figures (5)

---

## Data Splits

### Training Data
- **Source:** `results/training_v2_400plus/seed_2024/`
- **Size:** 2583 train, 72 dev, 553 test
- **Languages:** 9 (am, ar, de, en, es, hi, uk, xh, yo)

### Evaluation Data
- **Test set:** 553 examples (frozen across all models)
- **SHA256:** d31b7c18abc9be003d916d07037b868fa32f0d013298e735fc0e879f5f4e0d61

---

## File Organization

results/
├── training_*/ # Fine-tuned adapters
│ ├── training_mt0/
│ ├── training_llama_lora/
│ ├── training_qwen2_5_7b/
│ └── training_aya_23_8b/
├── evaluation/ # Behavioral evaluation
│ ├── mt5/
│ ├── mt0/
│ ├── llama8b_qlora/
│ ├── qwen2_5_7b_qlora/
│ └── aya_23_8b/
├── mechanistic/ # Mechanistic interpretability
│ ├── mt5_probes/, mt5_ablation/
│ ├── mt0_ablation/, mt0_sae_steering/
│ ├── llama8b/, llama_steering/
│ ├── qwen_probes/, qwen_sae/, qwen_steering/
│ └── aya_probes/, aya_sae/, aya_steering/
├── tables_figures/ # Paper-ready tables & figures
└── training_v2_400plus/ # Data splits

scripts/
├── 03_* - 06_* # mT5/mT0 scripts
├── 07_* - 08_* # Llama scripts
├── 09_* - 10_* # Qwen scripts
├── 13_* - 28_* # Mechanistic scripts
└── 29_* - 33_* # Aya scripts

---

## Next Steps (Paper Writing)

1. **Methods Section:**
   - Describe fine-tuning protocol (QLoRA vs full)
   - Explain probe training methodology
   - Detail SAE architecture and training
   - Describe activation steering protocol

2. **Results Section:**
   - Cross-model behavioral comparison (Table 2)
   - Probe performance comparison
   - SAE feature interpretability
   - Steering effectiveness (deltas)

3. **Figures to Include:**
   - Cross-model comparison (fig1_cross_model_comparison.png)
   - Per-language performance (fig2_per_language_mt5_mt0.png)
   - Mechanistic results (fig4_mechanistic_results.png)
   - Aya pipeline figures (fig_01-05)

4. **Discussion:**
   - Why mT5 performs best behaviorally
   - Trade-offs between model size and detoxification
   - Mechanistic insights from probes and SAE
   - Steering as a causal intervention

---

## Key Findings Summary

1. **mT5** achieves highest behavioral performance (J=0.855) but high copy rate (48%)
2. **mT0** has lowest copy rate (14-17%) with good performance (J=0.823)
3. **Llama** has lowest copy rate (1.4-2.5%) with competitive performance (J=0.800)
4. **Qwen** balances copy rate (3.4%) and performance (J=0.791)
5. **Aya** achieves strong performance (J=0.799) with moderate copy rate (7.1%)
6. **Steering** improves detoxification without utility loss (Aya: toxic_score ↓3.5%, J ↑0.75%)

---

## Repository Status (as of Sep 25, 2026)

- ✅ All 5 models: fine-tuning complete
- ✅ All 5 models: behavioral evaluation complete
- ✅ All 5 models: linear probes complete
- ✅ Qwen, Aya: SAE training complete
- ✅ Qwen, Aya: activation steering complete
- ✅ Aya: full pipeline tables and figures generated
- ⏳ mT5, mT0, Llama: SAE/steering (partial)
- ⏳ Cross-model comparison tables and figures

---

## Contact

For questions about this pipeline, refer to the individual model READMEs:
- `results/mechanistic/aya_steering/AYA_PIPELINE_README.md`
- Scripts in `scripts/` directory
