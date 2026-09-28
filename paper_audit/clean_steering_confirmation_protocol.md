# Clean steering confirmation protocol

## Goal
Confirm whether train-derived activation directions produce paired, test-set output changes
for multilingual detoxification while preserving content and output quality.

## Primary model
Aya-23-8B-Instruct, fine-tuned adapters for seeds 42 and 1337.

## Data separation
- Direction estimation: training split only.
- Layer/strength selection: development split only.
- Final evaluation: frozen 553-example test split only.
- No vector estimation, layer/strength selection, or qualitative selection on test data.

## Target languages
- Primary low-resource targets: Yoruba (`yo`) and isiXhosa (`xh`).
- Comparison high-resource source/target: English (`en`).
- Report all nine languages as secondary analysis only if compute permits.

## Direction conditions
1. No intervention baseline.
2. Within-language direction: yo→yo and xh→xh.
3. English transfer direction: en→yo and en→xh.
4. Matched-norm random direction.
5. Shuffled-pair direction: toxic/detox pairs randomly mismatched before averaging.

## Selection
- Candidate layers: {12, 16, 20, 24}.
- Candidate strengths: {-2, -1, -0.5, 0, 0.5, 1, 2}.
- Select one configuration using a predeclared development objective.
- Do not use the test set for selection.
- Retain the complete development sweep CSV with layer, strength, language, condition,
  all metrics, and selection objective.

## Test outputs
For every test example and condition retain:
- pair_id, language, toxic_input, detox_output;
- condition, vector_source_language, target_language, layer, strength;
- prediction;
- toxicity score or human non-toxicity rating;
- semantic similarity;
- reference-overlap chrF;
- copy flag;
- empty-output flag;
- output length;
- duplicate-output indicator.

## Primary comparison
For each low-resource target language, compare within-language and English-transfer
directions against paired no-intervention baseline on exactly the same frozen examples.

## Inference
- Paired bootstrap 95% confidence intervals for all predeclared metric differences.
- Report complete per-language counts.
- Report all tested conditions, not only selected conditions.

## Claim gate
A steering effect may enter the main paper only if:
- direction estimation uses train data only;
- selection uses dev data only;
- paired test outputs are available;
- a matched random and shuffled-pair control are available;
- non-toxicity outcome is documented as valid for the language or evaluated by humans;
- semantic preservation and output-quality controls are not materially degraded;
- the result replicates in a second seed or model.
