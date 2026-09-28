# Probe-results table provenance

## Source artifacts
- `results/mechanistic/aya_probes/seed_42/probe_results.json`
- `results/mechanistic/aya_probes/seed_1337/probe_results.json`
- `results/mechanistic/qwen_probes/seed_42/probe_results.json`
- `results/mechanistic/qwen_probes/seed_1337/probe_results.json`
- `results/mechanistic/qwen_probes/seed_2025/probe_results.json`

## Generated files
- `table_probe_results_long.csv`
- `table_probe_results_layer20_summary.csv`

## Protocol labels
- `auc_all`: pooled random-example split, stratified by binary toxic-versus-detox label.
- `auc_high_resource`: same split procedure restricted to
  `{am, ar, de, en, es, hi, uk}`.
- `auc_hr_to_lr`: train probe on all examples from
  `{am, ar, de, en, es, hi, uk}` and evaluate on `{xh, yo}`.

## Limits
Random-example splitting is not grouped by original pair or language. Therefore,
paired toxic/detox examples may appear on opposite sides of random splits. Probe AUC
is evidence of representation decodability under this protocol, not causal control,
independent toxicity measurement, or universal cross-lingual transfer.
