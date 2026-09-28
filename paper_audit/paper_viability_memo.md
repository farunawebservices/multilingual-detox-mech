# ARR long-paper viability memo

## Proposed title
Decodable but Not Yet Controllable: Auditing Multilingual Detoxification Representations

## Submission type
ARR long paper: empirical representation study plus reproducibility and evaluation audit.

## One-sentence thesis
Across multi-seed Aya-23-8B and Qwen2.5-7B-Instruct runs, toxic-input versus
detoxified-target representations are linearly decodable under the stored probe
protocol, but transfer from seven designated high-resource languages to isiXhosa and
Yoruba is substantially weaker. Recovered steering sweeps show strength-dependent
generation changes, including a non-monotonic Amharic reference-overlap profile; however,
the archived intervention artifacts do not by themselves establish validated multilingual
toxicity reduction or general causal controllability.

## Completed main-text evidence
1. Aya layer-20 probe results across seeds 42 and 1337.
2. Qwen layer-10/15/20 probe results across seeds 42, 1337, and 2025.
3. Generated layer-20 summary table and two figures derived from frozen probe JSONs.
4. Audited language partition:
   HR={am, ar, de, en, es, hi, uk}; LR={xh, yo}.
5. Verified evaluator limitation:
   STA is constant 1.0 and legacy J=(1+SIM+chrF)/3.
6. Verified Aya layer-20 zero-strength null control:
   553 paired test predictions are byte-identical.
7. Audited exclusions:
   Qwen steering uses test data to construct its vector and has no paired test baseline;
   Llama steering is estimated; Aya L24 sweep lacks required raw records.

## Main claims
C1. Toxic-input and detoxified-target representations are linearly decodable under
    the stored random-example probe protocol in Aya and Qwen.
C2. HR-to-LR transfer to isiXhosa and Yoruba is lower than pooled random-example AUC
    for both model families.
C3. Decodability alone does not establish causal controllability.
C4. The project’s legacy behavioral J cannot support toxicity or safety claims.

## Explicit non-claims
- No validated multilingual toxicity reduction from the archived steering artifacts.
- No behavioral model ranking by legacy J.
- No universal cross-lingual conclusion.
- No general causal detoxification-circuit claim.
- No confirmatory Qwen, Llama, or Aya-L24 steering effect claim.
- Exploratory mT5/MT0/cross-lingual steering results are reported with their data-leakage,
  baseline, metric, and replication limitations.

## Reviewer value
The paper contributes both original multi-seed probe evidence and a concrete case study
showing why representation, behavioral, and causal claims must be separated in
multilingual model analysis.

## Main-paper artifacts
- Table 1: multi-seed layer-20 probe AUC summary.
- Figure 1: pooled versus HR-to-LR transfer AUC.
- Figure 2: Qwen layer profile.
- Compact intervention and metric audit summary.

## Risks to address before submission
1. Establish novelty against prior multilingual detoxification and mechanistic-probing work.
2. Cite the dataset and source-model licenses correctly.
3. Explain that random splits are not pair-grouped in main text.
4. Complete required Limitations and Responsible NLP Checklist accurately.
5. Prepare fully anonymized supplementary materials.
6. Disclose AI writing/coding assistance in the acknowledgement and checklist.
