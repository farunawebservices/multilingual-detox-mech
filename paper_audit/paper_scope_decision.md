# Paper scope decision

## Central claim
Fine-tuned multilingual models can contain linearly decodable toxic-versus-detox
information, but decodability is not equivalent to causal controllability, and
cross-resource transfer is limited and heterogeneous.

## Include in main results
- Aya layer-20 probe AUC across seeds 42 and 1337.
- Qwen layer-20 probe AUC across seeds 42, 1337, and 2025.
- High-resource-to-low-resource probe transfer, explicitly defined as training on
  {am, ar, de, en, es, hi, uk} and testing on {xh, yo}.
- Metric audit showing constant STA and legacy J limitations.
- Aya layer-20 zero-strength steering null control.

## Appendix or reproducibility package
- Aya seed-42 descriptive SAE candidate feature ranking.
- Qwen seed-specific SAE candidate feature rankings.
- Training manifests and split hashes.
- Full claim registry and audit logs.
- Human-evaluation protocol and samples.

## Exclude from confirmatory claims
- All Qwen steering effects.
- Aya layer-24/-1.0 detox sweep.
- Any toxicity or safety improvement inferred from legacy J.
- Any universal causal-circuit claim.
- Behavioral model ranking based solely on J.

## Required language
Use "representation decodability," "reference-overlap proxy," and
"high-resource-to-low-resource transfer." Avoid "toxicity reduction,"
"validated safety," "fluency improvement," and "causal circuit" unless new
auditable evidence supports them.

## Supplementary-only result
- One mT5 teacher-forced decoder-state probe peaks at AUC 0.9442 at layer 8 on an
  unseeded random held-out-example split. It is exploratory because split IDs,
  random seed, group-wise pair separation, and seed replication are unavailable.

## Additional exclusions
- MT5 v1 probe results are not comparable to MT5 v2 and are excluded.
- MT5/MT0 head ablations are random-head sensitivity checks, not targeted
  mechanistic ablations.
- Llama probe results are in-sample; Llama steering values are estimated proxies.
