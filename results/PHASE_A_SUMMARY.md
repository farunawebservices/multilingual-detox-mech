# Phase A Complete: Mechanistic Analysis (mT5/mT0)

## Completed Experiments

### Stage 10 Part A: Linear Probes
- ✅ Decoder layers 5-7 best distinguish toxic vs detox (AUC 0.95-0.97)
- ✅ Encoder layers show no linear decodability
- ✅ Shared across all 9 languages

### Stage 10 Part B: Causal Ablation (mT5)
- ✅ Ablating 10/12 heads in layers 5-7: copy rate -8.7%
- ✅ Trade-off: SIM -5.5%, FL -2.6%, J -2.7%

### Stage 11: Causal Ablation (mT0)
- ✅ Replicates mT5 finding: copy rate -3.6%
- ✅ Same causal mechanism across encoder-decoder models

### Stage 10 Part C: Cross-Lingual Analysis
- ✅ High-resource probe AUC: 0.96-0.97 (shared circuits)
- ✅ Cross-lingual ablation: yo/xh copy rate 2.3% (transfer works)

## RQ Answers Summary

| RQ | Answer | Evidence |
|----|--------|----------|
| RQ1: Shared circuits | YES ✅ | Probe AUC 0.95-0.97 across all languages |
| RQ2: Cross-lingual transfer | YES ✅ | yo/xh copy 2.3% after English-identified ablation |
| RQ3: Causal steering | PARTIAL ⚠️ | Ablation shows causal effect (not steering yet) |
| RQ4: Architecture-specific | YES ✅ | Decoder only, not encoder |
| RQ5: Causal mechanisms | YES ✅ | mT5 -8.7%, mT0 -3.6% copy rate |
| RQ6: Cross-lingual intervention | YES ✅ | English→yo/xh transfer demonstrated |

## Confirmatory Claims

| Claim | Status |
|-------|--------|
| C1: Expanded-data improvement | ✅ Confirmed (mT5 -64%, mT0 -76% copy) |
| C2: Reduced collapse (yo/xh) | ✅ Confirmed (yo -24%, xh -21% copy) |
| C3: Cross-model robustness | ⚠️ Partial (mT5, mT0 confirmed; Llama pending) |
| C4: Causal quality criterion | ✅ Confirmed (causal effect exists with SIM trade-off) |

## Next Phase: Decoder-Only Models (Stage 6-7)
- Llama-3-8B zero/few-shot baseline
- Llama-3-8B LoRA fine-tuning
- Compare vs mT5/mT0
