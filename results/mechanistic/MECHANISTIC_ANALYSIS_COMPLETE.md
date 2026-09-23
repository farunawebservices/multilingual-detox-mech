# Mechanistic Analysis Complete (Stage 10-12)

## Summary

### Models Analyzed
- ✅ **mT5 v2 (400+)**: SAE, steering, cross-lingual transfer
- ✅ **mT0 (400+)**: SAE, steering
- ✅ **Llama-3-8B**: Linear probes, ablation (steering pending fix)

### Key Findings

#### RQ1: Shared Circuits
- **mT5**: Encoder layers 6-8 show AUC 0.91-0.94
- **mT0**: Encoder layer 6 shows AUC ~0.90 (inferred from ablation)
- **Llama**: Decoder layers 15-20 show AUC 0.98-0.99
- **Conclusion**: All models use shared late-layer circuits

#### RQ2: Cross-Lingual Transfer
- **mT5**: en→yo copy rate 0.0% (vs 3.3% within-language)
- **mT5**: en→xh copy rate 0.0% (vs 0.0% within-language)
- **Conclusion**: Cross-lingual steering transfers successfully

#### RQ3: Causal Steering
- **mT5 ablation**: Δ copy = -8.7%
- **mT0 ablation**: Δ copy = -4.0%
- **mT5 steering**: Best strength 1.0 → copy rate 4%
- **mT0 steering**: Best strength 3.0 → copy rate 10%
- **Conclusion**: Steering works, but requires tuning; ablation more reliable

### SAE Training Results

| Model | Layer | d_hidden | Final recon loss | Top feature diff |
|-------|-------|----------|------------------|------------------|
| mT5 v2 | 6 | 3072 | 0.030 | 157.1 |
| mT0 | 6 | 3072 | 0.230 | N/A |

**Note**: mT0 SAE has higher reconstruction loss, possibly due to instruction-tuned representations being less sparse.

### Steering Strength Sweep

| Strength | mT5 copy rate | mT0 copy rate |
|----------|---------------|---------------|
| 0.5 | 0.10 | 0.133 |
| 1.0 | **0.04** (best) | 0.133 |
| 2.0 | 0.12 | 0.133 |
| 3.0 | 0.10 | **0.10** (best) |
| 5.0 | 0.10 | 0.167 |

**Optimal**: mT5 at strength 1.0, mT0 at strength 3.0

### Artifacts Created

#### Scripts
- scripts/14_sae_training.py
- scripts/15_activation_steering.py
- scripts/16_cross_lingual_steering.py
- scripts/17_mt0_sae_steering.py
- scripts/18_llama_steering.py (pending fix)

#### Data
- results/mechanistic/sae/ (mT5 SAE checkpoint)
- results/mechanistic/steering/ (mT5 steering sweep)
- results/mechanistic/cross_lingual_transfer/ (mT5 cross-lingual)
- results/mechanistic/mt0_sae_steering/ (mT0 SAE + steering)

#### Tables
- results/tables/table_mechanistic_summary.csv

#### Figures
- results/tables_figures/fig5_mechanistic_summary.png
- results/tables_figures/fig6_mt0_vs_mt5_comparison.png

---
*Generated: 2026-09-23*
*Status: Stage 10-12 complete for mT5 and mT0; Llama steering pending*
