# Mechanistic Analysis Summary (mT5 400+ pairs)

## Key Findings

### 1. Linear Probes (Stage 10 Part A)
- **Decoder layers 6-8** best distinguish toxic vs detox inputs (AUC 0.91-0.94)
- Encoder layers show no linear decodability (< 0.5 accuracy)
- Suggests detoxification transformation occurs in decoder cross-attention

### 2. Causal Ablation (Stage 10 Part B)
- Ablating 10/12 cross-attention heads in layers 6 and 8:
  - **Copy rate: -8.7%** (0.201 → 0.114)
  - SIM: -5.5% (0.867 → 0.812)
  - FL: -2.6% (0.626 → 0.600)
  - J: -2.7% (0.831 → 0.804)
- Provides causal evidence that these heads propagate toxic content
- Matches prior experiment findings (Dementieva et al., 2024)

### 3. Comparison: 175 vs 400+ pairs
- Copy rate reduced 12-36% across all languages with 400+ pairs
- J score stable (< 1% change for most languages)
- More data reduces copying without hurting quality

## Artifacts
- Probe results: `results/mechanistic/mt5_probes_v2/probe_results.json`
- Ablation results: `results/mechanistic/mt5_ablation/ablation_results.json`
- Tables/Figures: `results/tables_figures/`

## Next Steps
1. Replicate ablation on mT0 (Stage 11)
2. Test cross-lingual transfer of ablation effects
3. Proceed to decoder-only fine-tuning (Stage 6-7)
