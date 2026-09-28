# Paper outline: audit-aware representation decodability

## Working title
From Decodability to Control: An Audit of Multilingual Detoxification Representations

## Abstract
- Problem: multilingual detoxification claims often combine behavioral proxies,
  representation probes, and interventions without equivalent evidential strength.
- Method: forensic audit of frozen artifacts from multilingual fine-tuning pipelines.
- Main result: Aya and Qwen show multi-seed toxic-versus-detox decodability, but
  high-resource-to-isiXhosa/Yoruba transfer is about 0.24 AUC lower.
- Audit result: legacy J cannot measure toxicity because STA is constant; available
  steering evidence does not support causal effect claims.
- Contribution: a bounded representation finding plus a reproducibility/evaluation audit.

## 1 Introduction
- Distinguish decodability, causal control, and behavioral safety.
- State the audit-aware research question.
- Contributions: audited evidence ledger, multi-seed probes, transfer gap, metric and
  intervention audit.

## 2 Data, Models, and Audit Protocol
- Nine-language paired corpus and known frozen split provenance.
- Aya and Qwen primary models/seeds.
- Evidence hierarchy and claim registry.
- Explicit language partition for transfer analysis.

## 3 Representation Probes
- Representation construction and logistic regression protocol.
- Random-example split caveat.
- Table 1: layer-20 multi-seed probe summary.
- Figure 1: pooled versus HR-to-LR transfer.
- Figure 2: Qwen layer profile.

## 4 Intervention and Metric Audit
- Constant-STA legacy J formulation.
- Aya zero-strength null control.
- Exclusion criteria for Qwen and Llama steering.
- MT5/MT0 random-ablation status.

## 5 Discussion and Limitations
- Decodability is not causal control.
- Pair leakage risk.
- Restricted two-language transfer target.
- Missing/recovered artifact constraints.
- Need for group-wise held-out protocols, validated multilingual toxicity assessments,
  and paired baseline-versus-intervention tests.

## 6 Conclusion
- State only the audited finding and the main methodological lesson.

## Appendix
- Full seed-level probe table.
- mT5 exploratory probe.
- SAE descriptive inventories.
- Intervention-audit table.
- Metric audit table.
- Claim registry and source manifests.
