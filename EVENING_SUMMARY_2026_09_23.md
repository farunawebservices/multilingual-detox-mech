# Evening Summary: September 23, 2026
## Mechanistic Analysis Complete (Stage 10-12)

### What We Accomplished

#### 1. Cross-Model Evaluation (Stage 8) ✅
- **mT5 v2 (400+)**: Copy 18.7%, SIM 0.856, FL 0.611, J 0.822
- **mT0 (400+)**: Copy 15.6%, SIM 0.845, FL 0.610, J 0.818
- **Llama-3-8B QLoRA**: Copy 2.2%, SIM 0.787, FL 0.599, J 0.795
- **Winner**: Llama-3-8B achieves lowest copy rate (2.2%) with competitive SIM/FL

#### 2. Mechanistic Analysis - mT5 (Stage 10-12) ✅
- **Linear probes**: AUC 0.91-0.94 at layers 6-8 (shared circuits)
- **SAE training**: 4x overcomplete (d_hidden=3072), recon loss 0.030
- **Top SAE feature**: diff=157.1 (detox > toxic)
- **Activation steering**: Best strength 1.0 → copy rate 4%
- **Cross-lingual transfer**: en→yo copy 0.0%, en→xh copy 0.0%

#### 3. Mechanistic Analysis - mT0 (Stage 10-12) ✅
- **Linear probes**: AUC ~0.90 at layer 6 (shared circuits replicated)
- **SAE training**: 4x overcomplete, recon loss 0.230 (higher than mT5)
- **Activation steering**: Best strength 3.0 → copy rate 10%
- **Ablation**: Δ copy = -4.0% (causal effect confirmed)

#### 4. Mechanistic Analysis - Llama-3-8B (Stage 10-12 partial) ✅
- **Linear probes**: AUC 0.98-0.99 at layers 15-20 (shared circuits)
- **Ablation**: yo/xh copy rate 4.0% (cross-lingual transfer)
- **Steering**: Pending fix (hook dimension mismatch)

### Research Questions Answered

| RQ | Question | Answer | Evidence |
|----|----------|--------|----------|
| RQ1 | Shared circuits? | ✅ YES | mT5 AUC 0.91-0.94, mT0 AUC ~0.90, Llama AUC 0.98-0.99 |
| RQ2 | Cross-lingual transfer? | ✅ YES | en→yo/xh copy 0-4% after high-resource ablation |
| RQ3 | Causal steering? | ✅ YES | mT5 Δ copy -8.7%, mT0 Δ copy -4.0%, steering works with tuning |

### Artifacts Created

#### Tables (8 total)
1. table_1_cross_model_comparison.csv
2. table_2_final_comparison.csv
3. table_3_metrics_comparison.csv
4. table_llama8b_detailed.csv
5. table_per_language_mt5_mt0.csv
6. table_summary_metrics.csv
7. table_mechanistic_summary.csv
8. table_dataset_stats.csv

#### Figures (6 total)
1. fig1_cross_model_comparison.png (4-panel bar charts)
2. fig2_per_language_mt5_mt0.png (3-panel per-language)
3. fig3_copy_vs_sim.png (scatter plot)
4. fig4_mechanistic_results.png (probe + ablation)
5. fig5_mechanistic_summary.png (4-panel mechanistic)
6. fig6_mt0_vs_mt5_comparison.png (mT0 vs mT5)

#### Scripts (18 total)
- Stage 1-8: Training and evaluation scripts
- Stage 10-12: Mechanistic analysis scripts (14-18)

#### Summary Documents
- results/PAPER_READY_SUMMARY.md
- CHECKLIST_STAGE13.md
- results/mechanistic/MECHANISTIC_ANALYSIS_COMPLETE.md
- EVENING_SUMMARY_2026_09_23.md

### Key Numbers for Paper

| Metric | mT5 (175) | mT5 v2 (400+) | mT0 (400+) | Llama-3-8B |
|--------|-----------|---------------|------------|------------|
| Copy rate | 0.49 | 0.187 ± 0.018 | 0.156 ± 0.016 | **0.022 ± 0.006** |
| SIM | 0.94 | 0.856 ± 0.012 | 0.845 ± 0.012 | 0.787 ± 0.001 |
| FL | 0.61 | 0.611 ± 0.012 | 0.610 ± 0.011 | 0.599 ± 0.019 |
| J score | 0.85 | 0.822 ± 0.008 | 0.818 ± 0.007 | 0.795 ± 0.007 |
| Probe AUC (best) | 0.944 | 0.944 | ~0.90 | 0.996 |
| Steering best | - | 4% @ 1.0 | 10% @ 3.0 | Pending |

### What's Next

1. **Fix Llama-3-8B steering** (hook dimension issue)
2. **Human evaluation** (Stage 9) - sample outputs, recruit annotators
3. **Paper writing** - Methods, Results, Discussion
4. **Supplementary materials** - Extended tables, examples

### Git Status
- Branch: main
- Commits tonight: ~10
- All artifacts: Saved and pushed to GitHub
- Environment: Pinned (requirements-lock.txt)

---
*Summary generated: 2026-09-23 20:45*
*Status: Stage 8 and Stage 10-12 complete for mT5 and mT0; ready for paper writing*
