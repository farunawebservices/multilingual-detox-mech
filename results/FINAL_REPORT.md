# Cross-Lingual Mechanistic Circuits for Text Detoxification in mT5

### A nine-language study with Yoruba and isiXhosa, in which the causal evidence is narrow and most of the interesting results are negative

---

## Abstract

We fine-tune `google/mt5-base` on parallel toxic→detoxified text in nine languages — Amharic, Arabic, German, English, Spanish, Hindi, Ukrainian, Yoruba and isiXhosa — and ask which internal components implement the transformation, and whether those components transfer to the two low-resource African languages. Working from 175 pairs per language (122 train / 27 dev / 26 test), we run linear probes, sparse autoencoders, gradient attribution, causal ablation, activation steering and cross-lingual steering transfer, each across three model seeds with matched random controls.

Three findings stand. First, toxic-versus-reference information is **strongly and reliably decodable** from decoder layer 6 (probe accuracy 0.933 ± 0.001 against a 0.553 confound baseline), and **mean-ablating the ten top-ranked decoder cross-attention heads causally raises measured non-toxicity** (ΔSTA +0.091, paired 95% CI [+0.038, +0.146]) while matched random heads do not (−0.011, CI [−0.044, +0.023]). Second, the same ablation **significantly degrades content preservation** (ΔSIM −0.066, CI [−0.084, −0.050]), so the mechanism is better described as a *toxic-content-propagation pathway* than as a detoxification circuit. Third, essentially everything else is negative: **no sparse-autoencoder feature survived replication** (0 of 600 candidates; under 1.2% of features match across seeds above cosine 0.7), **activation steering was null** at the pre-specified strengths (all eight frozen configurations have ΔJ intervals containing zero), and **cross-lingual steering transfer performed no better than a matched random vector of equal norm**.

For Yoruba and isiXhosa we report no validated toxicity numbers at all: the standard multilingual toxicity classifier is fine-tuned on nine languages that do not include either, so their STA is reported only as explicitly unvalidated and never enters a delta, an effect size, or a conclusion. Their behaviour is characterised instead through similarity, fluency and degeneracy diagnostics, which show a distinct and severe failure mode — isiXhosa collapses onto a single output string covering 58% of the test split, and Yoruba onto 67% unique outputs.

---

## 1. Research questions and hypotheses

**RQ1 — circuit structure.** Do multilingual models use shared or language-specific internal features when transforming toxic text into detoxified text?
**H1.** Some features/components are shared across languages; others are language-specific. Shared features transfer better between languages of similar representation quality.

**RQ2 — cross-lingual transfer.** Do circuits identified in high-resource languages transfer to Yoruba and isiXhosa?
**H2.** Transfer improves detoxification in Yoruba/isiXhosa but less than within-language steering, because of vocabulary, morphology, cultural and training-data differences.

**RQ3 — causal steering.** Can activation steering improve detoxification while preserving content?
**H3.** Steering directions from probes or SAE features improve toxicity reduction, with possible over-steering harming semantic similarity or fluency.

Answers are in §12–§14. In short: **H1 partially supported, H2 not supported under the method and strength range tested, H3 not supported as a general claim.**

---

## 2. Data, languages and splits

Nine languages, two corpora of origin:

| Source | Languages | Rows each |
|---|---|---|
| MultiParaDetox | am, ar, de, en, es, hi, uk | 400 |
| Human-annotated | **yo, xh** | 178 |

A larger supplementary English corpus (`en_detox_full.csv`, 19,744 rows) exists in the repository and was **never used as a tenth language**; it is registered separately in the configuration and excluded from every per-language loop.

### The 178 → 175 pair cap

The study caps every language at the size of the smallest, so that no result is driven by data volume. The nominal cap is the yo/xh corpus size of **178**. Yoruba, however, contains **3 exact duplicate (toxic, detox) pairs**, leaving **175 unique pairs**. The cap was therefore set at **175**, split **122 train / 27 dev / 26 test**, and every one of the nine languages hits those counts exactly.

Splitting rules, all enforced by assertions before any file was written:

- Rows sharing a normalised toxic input form a **group** assigned to exactly one split, so an input can never straddle partitions. Yoruba has 8 repeated toxic inputs, 6 of them with distinct references.
- Multi-reference pairs are preserved (a repeated input with two different human rewrites stays as two rows in the same split).
- Text is NFC-normalised **in the derived split data only**; the source CSVs were never modified. Yoruba mixed Unicode normalisation forms in 25 strings, so this matters for tokenisation consistency.
- Unused rows go to `reserve_<lang>.csv` at **group** granularity, so a later scaling experiment drawing on reserve cannot leak a test input into training.
- Verified: 0 pairwise `toxic_input` overlaps across all 12 partitions, 0 duplicate pairs, 0 test content in train or dev.

### Corpus asymmetries that matter later

| | am | ar | de | en | es | hi | uk | **yo** | **xh** |
|---|---|---|---|---|---|---|---|---|---|
| mT5 fertility (subwords/word) | 3.30 | 2.29 | 2.06 | 1.60 | 1.86 | 2.12 | 2.46 | **2.90** | **3.37** |
| reference/input length ratio | 0.79 | 0.91 | 0.93 | 0.91 | 0.93 | 0.91 | 0.95 | **1.43** | **1.24** |
| distinct leading bigrams | 0.97 | 0.96 | 0.95 | 0.90 | 0.96 | 0.96 | 0.98 | **0.48** | **0.75** |

Two of these are confounds rather than curiosities. The **inverted length ratio** means the yo/xh annotators *expanded* toxic inputs into longer rewrites while the MultiParaDetox annotators *deleted* toxic spans — the annotated task is not the same across the two corpora. And Yoruba's **48% leading-bigram diversity** means its 178 rows carry far less independent signal than the count implies.

One expected problem did **not** materialise: mT5's UNK rate is 0.0000 for both yo and xh, and round-trip fidelity is 99.4% for Yoruba. The tokenizer fragments Yoruba diacritics heavily but does not lose them.

---

## 3. Training setup

`google/mt5-base` (580M parameters, 12 encoder + 12 decoder blocks, d_model 768), fine-tuned on `data/splits/train.csv` (1,098 rows) with per-language dev loss on `data/splits/dev.csv` (243 rows), for seeds **42, 1337, 2024**.

| Setting | Value | Reason |
|---|---|---|
| Optimiser | AdamW, lr **1e-4**, wd 0.01, linear decay, 10% warmup | Conservative end of mT5's workable band. Lower risks the known failure where mT5 never leaves span-corruption pretraining and emits `<extra_id_0>` sentinels. |
| Precision | **bf16, never fp16** | mT5 overflows to NaN under fp16 (gated-GELU). A correctness requirement, not a speed choice. |
| Batch | 8 × 2 accumulation = **16 effective** | 69 optimiser steps/epoch. Update steps, not memory, are the binding constraint at this data size. |
| Lengths | source **96**, target **112** | Measured: train+dev source p99 = 73, corpus-wide target max = 102 (Amharic). Zero truncation anywhere. |
| Schedule | 20-epoch ceiling, early stopping patience 4 on aggregate dev loss | All three seeds stopped early (best epoch 8, 8, 9; ran 12, 12, 13), so the budget was not truncating a still-improving run. |
| Task prefix | `"detoxify: "`, **language-neutral** | A language-tagged prefix would plant an explicit language token that the Stage 7 language probes would simply read back. |

NaN/Inf audit passed for all three seeds on training loss, all 10 dev-loss values, and **every model parameter**. Note that `logging_nan_inf_filter` defaults to `True` in transformers v5 and would have silently hidden non-finite losses; it was disabled so the audit is real.

Per-language dev loss spans a 3.3× range and tracks tokenizer fertility closely: de 0.99, en 1.01, ar 1.03, hi 1.08, uk 1.18, es 1.27, **yo 2.08**, am 2.20, **xh 3.28**.

---

## 4. Evaluation metrics and the classifier-coverage limitation

Following the official TextDetox 2024 protocol:

- **STA** — style transfer accuracy, P(neutral) from `textdetox/xlmr-large-toxicity-classifier` (label mapping verified as `{0: neutral, 1: toxic}`).
- **SIM** — content preservation, LaBSE cosine between the toxic input and the generated output.
- **FL** — fluency, chrF1 against the human reference. The 2024 shared task used ChrF precisely because no CoLA-style fluency data exists for most of these languages.
- **J** — mean per-sample STA × SIM × FL.

### Yoruba and isiXhosa have no validated STA

The classifier's model card enumerates **nine fine-tuning languages: en, ru, uk, de, es, ar, am, hi, zh**. Seven of our nine are covered. **Yoruba and isiXhosa are not.** XLM-R pretraining includes xh but not yo, and neither has toxicity-labelled fine-tuning data, so the classifier still emits a confident-looking probability with no validated basis for it.

Consequently, throughout this project and this report:

- yo/xh STA appears **only** in columns explicitly suffixed `_unvalidated`;
- yo/xh **J is undefined** and reported as such;
- yo/xh STA **never enters** a delta, an effect size, a significance test, a selection decision, or a conclusion;
- yo/xh behaviour is characterised through **SIM, FL, copy rate, unique-output rate, top-output share and template rate**, all of which are valid for them (LaBSE declares 110 languages including yo and xh; chrF is language-agnostic).

### A second measurement problem, in the covered languages

Scoring the *inputs* revealed that the classifier rates **45.6% of known-toxic Amharic inputs and 43.3% of Arabic inputs as neutral**, against 0.001 for English. These are human-annotated toxic sentences. So a large part of the apparent STA for am/ar/de/es reflects weak classifier sensitivity rather than model behaviour, consistent with the card's own Amharic F1 of 0.778 — its lowest. We therefore also report **ΔSTA = STA(output) − STA(input)**, which reorders the results substantially.

---

## 5. Stage-by-stage method summary

| Stage | Method | Outcome class |
|---|---|---|
| 0–2 | Environment pinning, read-only corpus inventory, grouped deterministic splits | setup |
| 3 | mT5-base fine-tuning, 3 seeds, early stopping | setup |
| 4 | Beam-4 deterministic generation + degeneracy diagnostics | descriptive |
| 5 | STA/SIM/FL/J with coverage gating, paired bootstrap CIs | descriptive |
| 6 | Forward-hook activation collection, encoder/decoder layers {4,6,8}, 3 conditions | setup |
| 7 | Linear probes on standardised activations, cross-lingual transfer matrices | **associative** |
| 8–9 | TopK SAEs, language-balanced; stability-gated feature analysis | **negative** |
| 10 | Gradient×activation attribution over 10,552 components | **ranking, non-causal** |
| 11 | Mean ablation with matched random controls; dev then one test pass | **causal** |
| 12–13 | Steering-vector construction; dev strength sweep, frozen test | **null/weak** |
| 14 | Cross-lingual steering transfer, dev only | **null** |
| 15 | Paired bootstrap aggregation, evidence classification | statistics |
| 16–17 | Tables and figures from committed artifacts | reporting |

---

## 6. Baseline behaviour: two complementary failure modes

Beam-4 decoding, 3 seeds, 1,431 generations. **No empty outputs and no sentinel tokens anywhere** — mT5 adapted to the task format properly. But the model fails in two opposite, resource-correlated ways.

### Test-split baseline

| lang | STA | ΔSTA | SIM | FL | J | loose copy | unique out. | top-output share |
|---|---|---|---|---|---|---|---|---|
| am | 0.561 | +0.106 | 0.949 | 0.476 | 0.252 | 0.410 | 1.000 | 0.038 |
| ar | 0.514 | +0.080 | 0.976 | 0.696 | 0.355 | **0.500** | 1.000 | 0.038 |
| de | 0.602 | +0.210 | 0.973 | 0.813 | 0.473 | 0.410 | 1.000 | 0.038 |
| en | 0.584 | **+0.583** | 0.927 | 0.710 | 0.406 | 0.103 | 1.000 | 0.038 |
| es | 0.519 | +0.202 | 0.940 | 0.647 | 0.320 | 0.231 | 1.000 | 0.038 |
| hi | 0.333 | +0.196 | 0.894 | 0.634 | 0.228 | 0.397 | 1.000 | 0.038 |
| uk | 0.476 | +0.436 | 0.950 | 0.784 | 0.385 | 0.308 | 1.000 | 0.038 |
| **yo** | *unavailable* | — | **0.454** | **0.157** | — | 0.115 | **0.667** | **0.282** |
| **xh** | *unavailable* | — | **0.388** | **0.181** | — | 0.064 | **0.423** | **0.577** |

**Input copying dominates the seven comparison languages.** Roughly 30% of all generations are verbatim copies of the input — Arabic 50%, Amharic and German 41%, Hindi 40%. English is the only language below 20%. Copying preserves SIM perfectly (0.89–0.98) while performing no detoxification. Measured by ΔSTA rather than raw STA, Arabic (+0.080) and Amharic (+0.106) barely detoxify at all, while English (+0.583) and Ukrainian (+0.436) do so substantially.

**isiXhosa template collapse.** Only 42.3% of test outputs are unique, and **a single string accounts for 57.7% of them**. Worst case, seed 2024: 22 of 26 isiXhosa test outputs are the identical string `"Ndifuna ukuqinisekisa ukuba ulungile."`. SIM 0.388 and FL 0.181 confirm the output is fluent-looking but content-free.

**Yoruba repetition.** Milder but real: 66.7% unique outputs, top string covering 28.2%, e.g. `"Ìṣe rẹ nílò àtúnṣe"`. SIM 0.454, FL 0.157.

Two points of care. First, this is **not memorisation of training targets** — exact and diacritic-insensitive matches against training outputs are both ≈0. The model invents its own generic strings and collapses onto them. Second, the metric the brief specified for detecting this (`template_rate`, exact match against frequent training outputs) returned **0.000 for every language**, because across all nine languages exactly one training output repeats. What caught the collapse was the generation-side repetition statistics. We report both.

**Seed instability.** Only 40.3% (dev) / 31.2% (test) of pairs produce an identical string across all three seeds, despite near-identical dev loss (1.589–1.595). Any single-seed conclusion here would be unreliable.

---

## 7. Stage 7 — probes: associative evidence

Probes were fitted on train and evaluated on dev; **test activations were never read** (a path guard enforces this). All analyses run on activations standardised with train-only statistics, verified at max |dimension mean| = 2.4×10⁻⁶ and sd ∈ [0.9995, 0.9996] with zero constant dimensions.

Standardisation is not cosmetic here. mT5 carries massive outlier dimensions — per-dimension standard deviations span 16.8 to 62,838 — which drive raw decoder cosine similarity to 0.999 between *every* pair of conditions. Before standardisation the decoder looked least discriminative; after it, the decoder is the most discriminative site. Any shared-circuit claim resting on raw correlation would have measured a scale artefact.

### Condition probe (toxic vs reference), dev accuracy, 3 seeds

| side | layer | accuracy | control baseline |
|---|---|---|---|
| **decoder** | **6** | **0.933 ± 0.001** | 0.553 |
| decoder | 4 / 8 | 0.918 / 0.897 | 0.553 |
| encoder | 6 | 0.764 ± 0.008 | 0.553 |

The control baseline uses only character length, token count and language identity. It began as a *broken* control — computing features from `toxic_input` for both classes made them identical within a pair and pinned the baseline to exactly 0.500 by construction. Corrected to use each row's own condition text, it gives 0.553, and a per-language breakdown reveals the confound is **concentrated in exactly the two languages of interest**: length alone reaches **0.704 for Yoruba and 0.722 for isiXhosa**, against 0.500–0.593 elsewhere. This tracks their inverted length ratio.

**Language identity is perfectly decodable — 1.000 accuracy at every layer, side and pooling** (control 0.424), despite the deliberately language-neutral task prefix.

### Cross-lingual transfer of the probe (layer 6)

| source → target | decoder | encoder |
|---|---|---|
| within-language (yo / xh) | 1.000 / 1.000 | 0.920 / 0.802 |
| {EN,DE,ES} → yo / xh | 0.691 / 0.673 | **0.494 / 0.500** |
| all non-African → yo / xh | 0.827 / 0.698 | **0.537 / 0.506** |
| {YO,XH} → am … ar | 0.963 … 0.654 | 0.691 … 0.512 |

Decoder transfer into the African languages is above chance but well below within-language; **encoder transfer is at chance**. This is the clearest structural evidence for partial sharing — but it is *associative*, and the yo/xh within-language figure of 1.000 must be read against a 0.70–0.72 length-only baseline, not against 0.5.

---

## 8. Stages 8–9 — sparse autoencoders: a negative stability result

TopK SAEs (custom implementation; `sae-lens` was verified unusable — TransformerLens lists `google-t5/t5-{small,base,large}` but has no mT5 entry and cannot load a local fine-tuned encoder-decoder checkpoint). Multilingual SAEs sampled **equal token counts per language** (3,094 each), which matters because isiXhosa needs 3.37 subwords per word against English's 1.60.

**Reconstruction is adequate.** Encoder dev explained variance 0.962–0.989 with dead-feature rates 0.13–0.49; decoder 0.750–0.782 with **zero** dead features. Expansion factor 4 rather than the configured 8, because layer 6 offers ~36k tokens after balancing and 8× would leave under 6 tokens per latent.

**Feature stability is the problem.** Across seeds, mean matched decoder-direction cosine is 0.27–0.29 at layer 6, and the fraction of features finding a counterpart above cosine 0.7 is **1.14% (decoder) and 0.36% (encoder)**. The conclusion does not depend on the threshold: 18.4% at 0.5, 9.0% at 0.6, 3.7% at 0.7, 1.6% at 0.8.

**The stability-gated promotion funnel (Stage 9), 600 layer-6 candidates discovered on train:**

| criterion | passing |
|---|---|
| separates toxic vs reference on dev | 53 |
| survives language/length controls | 511 |
| predicts a behaviour distinction | 270 |
| replicates in ≥2 of 3 seeds | 26 |
| **all four → detoxification feature** | **0** |

**No SAE feature is called a detoxification feature.** Two supporting observations: candidate directions are nearly orthogonal to the probe direction that achieves 0.933 accuracy (mean |cosine| 0.016–0.079), suggesting the information is distributed rather than concentrated in single latents; and candidates are near chance on the behavioural tests (mean |AUC−0.5| of 0.040 for copy-vs-ordinary), which is what a generic text-property feature looks like.

---

## 9. Stage 10 — attribution: ranking, explicitly not causality

Gradient×activation attribution onto a decoder layer-6 readout, for the probe direction and the train-derived toxic-minus-reference mean difference. 10,552 components: 312 attention heads (encoder self-attention L0–11, decoder self- and cross-attention L0–6) plus 10,240 MLP neurons. Splits are language-balanced by construction (122 train / 27 dev rows per language per condition).

Decoder attention heads carry 2–4× the attribution of MLP neurons or encoder heads. Ranking is reasonably stable across seeds (Spearman ρ 0.663–0.685; top-25 overlap 18.3–21.0 of 25) — far more stable than the SAE features.

Matched random controls temper the head rankings: top-25 neurons clear the random 95th percentile **25/25**, but top heads only **≈16/25**, because the head null is heavy-tailed. Roughly a third of "top" heads are not distinguishable from a random head.

A **zero-path validity check caught a serious defect**. Decoder layers above the layer-6 readout have no gradient path and must show exactly zero attribution; they did not. The cause was reading activations in hook-dict insertion order — which is *forward-execution* order, placing encoder block 4's MLP between self-attention layers 4 and 5 — so every attribution column was mislabelled. After the fix, above-readout attribution is exactly 0.000000 and the below-readout maximum rose from 0.044 to 2.214. All reported numbers come from the corrected run.

**No component is called a circuit on this evidence.** Attribution is a first-order approximation to patching and measures association with a readout, not causal necessity.

---

## 10. Stage 11 — causal ablation: the one positive result

Mean ablation (replacing a component with its train-set mean) rather than zero ablation, because mT5 activations peak near 2×10⁵ and zeroing injects a large off-distribution shift that would conflate removing information with knocking the residual stream off its manifold. The design — arm membership, random-control matching by type, sublayer, layer and count — was **frozen to disk with a SHA-256 digest before any generation**; dev was evaluated first, then test **exactly once** under the verified digest, guarded by a stamp file that refuses a second run.

### Test split, macro-averaged over nine languages, 3 seeds, paired bootstrap 95% CI

| arm | ΔSTA [95% CI] | ΔSIM [95% CI] | d_z | verdict |
|---|---|---|---|---|
| **top_decoder_cross_heads** (10) | **+0.0909 [+0.0378, +0.1460]** | **−0.0659 [−0.0835, −0.0498]** | 0.236 | causal, **harmful trade-off** |
| **top_probe_direction** (25) | **+0.0855 [+0.0349, +0.1391]** | **−0.0643 [−0.0812, −0.0485]** | 0.225 | causal, **harmful trade-off** |
| **top_mean_diff_direction** (25) | **+0.0567 [+0.0110, +0.1059]** | **−0.0407 [−0.0562, −0.0261]** | 0.162 | causal, **harmful trade-off** |
| top_decoder_self_heads (10) | +0.0319 [−0.0024, +0.0688] | −0.0127 [−0.0239, −0.0021] | 0.124 | null |
| top_encoder_components (25) | +0.0197 [−0.0129, +0.0535] | −0.0181 [−0.0316, −0.0061] | 0.083 | null |
| **top_decoder_mlp_neurons** (50) | **−0.0216 [−0.0447, −0.0022]** | +0.0009 | −0.130 | **causal-harmful** |
| *random_heads_matched* (20) | −0.0106 [−0.0444, +0.0225] | −0.0023 | −0.043 | null ✓ |
| *random_neurons_matched* (50) | −0.0030 [−0.0166, +0.0105] | +0.0002 | −0.006 | null ✓ |
| *language_detectors* (25) | −0.0122 [−0.0461, +0.0201] | +0.0050 | −0.052 | null ✓ |
| *sae_replicated_features* (15) | +0.0034 [−0.0215, +0.0295] | −0.0035 | 0.023 | null ✓ |

**The control structure is what makes this interpretable.** All four control arms — matched random heads, matched random neurons, language-detector components, and the retained SAE arm — have intervals containing zero. Three primary arms exclude zero. Matched random heads drawn from the *same layers and sublayers* move STA in the *opposite* direction (−0.011) to the real heads (+0.091).

Direction of effect: ablation **increases** measured non-toxicity. These heads carry toxic content through to the output; removing them suppresses it.

### The SIM-collapse trade-off

Every arm that significantly raises STA also significantly lowers SIM. For the strongest arm, ΔJ is only +0.014 despite ΔSTA of +0.091. **The intervention is partly buying non-toxicity by discarding content**, which is precisely the failure mode the study protocol required us to check for and flag. It is flagged in the artifacts, in the tables and here.

### Language-specific effects, and why yo and xh are never merged

Test split, `top_decoder_cross_heads`:

| | am | ar | de | en | es | hi | uk | **yo** | **xh** |
|---|---|---|---|---|---|---|---|---|---|
| ΔSTA | +0.135 | +0.100 | +0.035 | +0.113 | **+0.287** | −0.017 | −0.016 | *unavailable* | *unavailable* |
| ΔSIM | **−0.170** | −0.039 | −0.039 | −0.040 | −0.086 | −0.023 | −0.027 | **−0.124** | −0.044 |
| Δ unique-output rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **−0.167** | **+0.077** |

The effect is **not uniform**. Hindi and Ukrainian show no STA gain at all. Amharic gains STA but loses more SIM than any other language. And the two African languages respond in **opposite directions on degeneracy**: ablation makes Yoruba *more* collapsed (unique-output rate −0.167, top-output share +0.167) while making isiXhosa slightly *less* collapsed (+0.077 / −0.115). Aggregating them into a single "African languages" result would erase a real qualitative difference, so we never do.

We make **no toxicity claim for Yoruba or isiXhosa here.** Their STA is unavailable; what we can say is that this ablation reduces Yoruba's content preservation and worsens its output diversity.

---

## 11. Stages 12–13 — steering vectors and inference-time steering: weak to null

**Stage 12** built 1,743 candidate directions from train activations only: 432 mean-difference and 432 probe vectors (primary), 864 matched random controls, and 15 SAE decoder rows retained strictly as a control arm. All L2-normalised to unit norm with pre-normalisation magnitude recorded.

Mean-difference vectors are markedly more reproducible than probe vectors — cross-seed cosine **0.981 vs 0.930** at decoder layer 6 (random controls 0.007). This matters because **360 of the 432 probe vectors are underdetermined**, fitted on 244 samples in 768 dimensions; each is flagged in the manifest.

Cross-language geometry is informative on its own: mean-difference vectors agree at cosine **0.798** among the seven comparison languages and **0.739** between Yoruba and isiXhosa, but only **0.635** across the group boundary (en–xh 0.51, en–yo 0.52). The toxic→detox direction is partially shared but measurably rotated for the African languages.

**Stage 13** swept strengths {−1.0, −0.5, 0.0, +0.25, +0.5, +1.0} over 21 arm/scope combinations × 3 seeds on dev (91,854 generations), selected a per-language configuration **using dev only**, froze it with a digest, and evaluated test **once**. For Yoruba and isiXhosa, selection deliberately **excluded STA** and used SIM × FL subject to degeneracy guardrails, with STA marked unavailable.

**All eight frozen test configurations are null.** Every ΔJ interval contains zero:

| configuration | ΔSTA [95% CI] | ΔJ [95% CI] |
|---|---|---|
| encoder/mean_diff/per_language@+0.50 | +0.018 [−0.014, +0.053] | +0.001 [−0.023, +0.024] |
| encoder/mean_diff/pooled_all_balanced@+0.50 | −0.003 [−0.034, +0.027] | +0.005 [−0.017, +0.028] |
| encoder/probe/pooled_all_balanced@+1.00 | 0.000 [−0.021, +0.022] | −0.003 [−0.019, +0.011] |
| *(five further configurations)* | all intervals contain zero | all intervals contain zero |

Decoder-side steering was **inert** — maximum |ΔSIM| of 0.005 across all cells at every strength. This sits oddly beside Stage 11, where *ablating* decoder heads produced large effects, and the most likely explanation is magnitude: a unit-norm rank-1 addition is a far smaller perturbation than removing a head's full 64-dimensional contribution. We treat this as a limitation of the tested strength range, not as evidence that the decoder is causally unimportant.

---

## 12. Stage 14 — cross-lingual transfer: null against random controls

Source-group vectors were built from **train only** for {EN, DE, ES}, all non-African languages, and {YO, XH}, then cross-tested on complementary targets on **dev only** — the Stage 13 test artifacts were neither read nor written. The {EN, DE, ES} group did not exist in the Stage 12 library and was constructed here; the two overlapping groups were rebuilt and verified against the stored vectors at **cosine 1.000000**, confirming an identical construction path.

**Into Yoruba and isiXhosa** (encoder, mean over seeds), SIM falls monotonically with strength — isiXhosa 0.541 at α=−1.0 → 0.386 at α=0 → 0.117 at α=+1.0. Crucially:

| into yo/xh | mean ΔSIM | worst ΔSIM |
|---|---|---|
| transfer vector (real) | −0.031 | −0.326 |
| **matched random control** | −0.047 | −0.160 |

And macro-averaged over the whole transfer sweep:

| family | ΔSTA | ΔSIM | ΔFL | ΔJ |
|---|---|---|---|---|
| transfer_primary | +0.0057 | −0.0090 | −0.0077 | −0.0042 |
| **transfer_random_control** | +0.0144 | **−0.0241** | **−0.0179** | −0.0133 |

**The matched random vector moves every metric more than the real transfer vector does.** There is no transfer advantage to detect. Degeneracy diagnostics confirm nothing improved: template rate stays 0.000, isiXhosa's unique-output rate stays ≈0.47, and Yoruba's loose-copy rate *rises* with strength (0.086 → 0.111). Transfer from {YO, XH} into the comparison languages is likewise indistinguishable from random to three decimals.

---

## 13. Stage 15 — statistical methodology and evidence classes

All intervals are **paired at the sentence-pair level**. Each `pair_id` is differenced against its own baseline under the same seed and language, and the bootstrap (1,000 resamples) resamples *pair_ids*, not rows. Resampling rows independently would break the pairing and misstate the correlation between an intervention and its baseline. Macro averages resample **stratified by language**, preserving the nine-language balance inside every replicate. Effect sizes are Cohen's **d_z**, the matched statistic for a paired design.

**An interval containing zero is reported as null, never as a trend, and no p-value is manufactured for a comparison that does not have one.** `top_decoder_self_heads` at ΔSTA +0.032 [−0.002, +0.069] is the fourth-largest effect and is labelled null.

| # | Evidence class | Finding | Status |
|---|---|---|---|
| 1 | **ASSOCIATIVE** | Probes separate toxic from reference (0.933 vs 0.553 control); decoder transfers to yo/xh above chance | correlational; no causal test |
| 2 | **NEGATIVE** | No SAE feature is a detoxification feature (0/600); 3.7% replicate | hypothesis tested, not supported |
| 3 | **CAUSAL** | Ablating top decoder cross-attention heads raises STA (+0.091, CI excludes zero); all controls null | causal, effect size modest (d_z 0.236) |
| 4 | **CAUSAL-HARMFUL** | The same ablation degrades SIM (−0.066, CI excludes zero); MLP-neuron ablation lowers STA | causal in an undesired direction |
| 5 | **NULL/WEAK** | Steering at unit norm: all frozen arms ΔJ contain zero | null at tested strengths |
| 6 | **NULL** | Cross-lingual transfer: random control moves metrics more | no advantage over arbitrary direction |

---

## 14. Answers to the research questions

### RQ1 / H1 — **partially supported**

Yes, there is a component set with a **causal** effect on toxicity in output that operates across languages: mean-ablating the ten top-ranked decoder cross-attention heads raises measured non-toxicity by +0.091 (CI [+0.038, +0.146]) where matched random heads from the same layers and sublayers do not (−0.011, CI [−0.044, +0.023]). This meets the study's bar of correlation **plus** a causal effect with matched controls, and it is the only finding in the project that does.

But three qualifications are essential, and we state them rather than bury them:

1. **The mechanism is toxic-content propagation, not detoxification.** Ablation *removes* something that carries toxic content forward. It does not engage a detoxification capability, and the significant SIM loss (−0.066) shows part of the gain is content being discarded.
2. **Language-specific structure is at least as prominent as sharing.** Language identity is perfectly linearly decodable (1.000) at every site; encoder probes transfer to yo/xh at chance; and the ablation effect is non-uniform, with no STA gain at all for Hindi or Ukrainian and opposite degeneracy effects for Yoruba and isiXhosa.
3. **No stable SAE feature was found.** The interpretable-feature route to H1 failed outright — 0 of 600 candidates passed replication, controls and behaviour together, and candidate directions are nearly orthogonal to the probe direction that works. The toxic/detox information appears distributed rather than concentrated in individual latents.

H1's second clause — that shared features transfer better between languages of similar representation quality — is **consistent with** the probe transfer pattern and the vector geometry (cosine 0.798 within the comparison group vs 0.635 across the group boundary), but that evidence is associative and partly confounded by the yo/xh annotation-style difference.

### RQ2 / H2 — **not supported** under the method and strength range tested

H2 predicted that transfer would improve detoxification in Yoruba and isiXhosa, but less than within-language steering. The transfer experiments do not support the antecedent: steering vectors built from {EN, DE, ES} or from all non-African languages produced **no benefit** for either African language, and performed **no better than a matched random vector of equal norm** — indeed the random control moved SIM, FL and J more.

Two honest caveats in the other direction. First, the null extends to the **within-language** condition too, so this is more plausibly a statement about the intervention being under-powered at unit norm than about transfer failing specifically. Second, we cannot evaluate the *toxicity* half of H2 for yo/xh at all, because their STA is unvalidated; the null is established on SIM, FL and degeneracy diagnostics.

What *does* survive as suggestive, associative evidence for partial transfer is the Stage 7 probe result — decoder probes trained on non-African languages classify yo/xh conditions at 0.67–0.83 against a within-language 1.00 — and the rotated-but-correlated vector geometry. Neither is a demonstration that a transferred circuit improves detoxification.

### RQ3 / H3 — **not supported as a general claim**

H3 predicted that steering directions from probes or SAE features would improve toxicity reduction, possibly at the cost of similarity or fluency. Neither half holds as stated:

- **Steering did not reliably improve J.** All eight frozen test configurations have ΔJ intervals containing zero. Decoder-side steering was inert at every strength; encoder-side steering at |α| = 1.0 degraded SIM and FL without a compensating STA gain. SAE-derived vectors, retained as a control arm, were null throughout — consistent with the Stage 9 negative result.
- **Ablation does change behaviour causally, but harms content preservation.** The one intervention that works reliably raises STA by +0.091 while lowering SIM by −0.066, yielding ΔJ of only +0.014. The over-steering harm H3 anticipated is real, but it arrives via ablation rather than steering, and it is not offset by a proportionate toxicity benefit.

So the answer to "can activation steering improve detoxification while preserving content?" is: **not with the methods and magnitudes tested here.** The single reliable causal lever we found trades content for toxicity reduction rather than improving both.

---

## 15. Limitations

1. **Small-data regime (175 pairs/language).** The cap equalises languages but leaves every per-language analysis underpowered. Test splits are 26 rows per language, so per-language confidence intervals are wide and most between-language differences are not separable.
2. **~122 training pairs per language.** 1,098 total training rows for a 580M-parameter model. The baseline is weak in specific, diagnosable ways, and every mechanistic finding describes *this fine-tune*, not detoxification in general.
3. **Classifier coverage gaps for Yoruba and isiXhosa.** The toxicity classifier is fine-tuned on nine languages, neither of which is yo or xh. **No validated toxicity number exists for either language anywhere in this study.** Their STA is reported only as unvalidated and never used for selection, deltas, effect sizes or conclusions. Additionally, within the covered languages, the classifier rates 45.6% of known-toxic Amharic and 43.3% of Arabic inputs as neutral, so STA is a weak instrument there too.
4. **No native-speaker validation.** No Yoruba or isiXhosa speaker reviewed any output. Every claim about those languages rests on automatic metrics — LaBSE similarity, chrF and repetition statistics — and none of these substitutes for human judgement of whether a rewrite is non-toxic, natural or culturally appropriate. This is the single most important caveat on the yo/xh results.
5. **Poor tokenizer fertility for several languages.** isiXhosa needs 3.37 subwords per word and Amharic 3.30, against English's 1.60. Fertility tracks dev loss closely (xh 3.28, am 2.20 vs de 0.99), so part of what looks like a representational finding is tokenisation efficiency.
6. **Output copying.** ~30% of baseline generations are verbatim input copies (Arabic 50%). Circuit analyses are therefore partly characterising a copy mechanism rather than a rewriting one.
7. **Template collapse.** isiXhosa produces 42.3% unique outputs with one string covering 57.7%; Yoruba 66.7% unique. Activations for these languages are low-entropy, which inflates apparent reconstruction quality and separability in ways that are symptoms, not mechanisms.
8. **Unbalanced annotation-style differences.** yo/xh references are *longer* than their inputs (ratio 1.43 / 1.24) while the other seven are shorter (0.79–0.95). The two corpora encode different annotation conventions — expansion versus deletion — so cross-group comparisons conflate representation with task definition. Length alone classifies the yo/xh condition contrast at 0.70–0.72.
9. **SAE instability.** Under 1.2% of features replicate across seeds above cosine 0.7 at 8–12 tokens per latent. The SAE arm is uninformative here; a better-determined SAE would need substantially more data, not better hyperparameters.
10. **Attribution is not causality.** Gradient×activation is a first-order approximation. The MLP-neuron arm ranked highly by attribution and produced a *significant effect in the opposite direction* under ablation. Attribution rankings are reported as rankings only.
11. **Harmful STA/SIM trade-off.** Every significant ablation arm buys STA at SIM's expense. J barely moves. No intervention in this study improves detoxification and content preservation simultaneously.
12. **Weak steering magnitude.** Steering used unit-norm vectors at |α| ≤ 1.0, fixed in advance. The decoder null and the transfer null are both consistent with the intervention being too small rather than the directions being wrong. This range was pre-specified and not extended, so the steering and transfer results are conditional on it.
13. **Test used exactly once.** Per protocol, the test split was evaluated once for the ablation stage and once for the frozen steering configuration, each under a digest-verified frozen design with a stamp file preventing re-runs. This is the correct discipline, but it means **no test-set result here has been replicated**, and dev effects were roughly double test effects for the strongest ablation arm (+0.175 dev vs +0.091 test).
14. **Model-specific scope.** Everything is `mt5-base` fine-tuned on 1,098 pairs. Nothing here should be assumed to hold for larger mT5 variants, other multilingual architectures, decoder-only models, or better-trained detoxification systems.
15. **Seed instability in outputs.** Only 31% of test pairs produce identical strings across seeds. Three seeds are the minimum for the claims made and are not sufficient for fine-grained per-language conclusions.

---

## 16. Reproducibility

Repository: `multilingual-detox-mech`, branch `detox-circuits-pipeline`. Environment: Python 3.12.13, torch 2.12.0+cu130, transformers 5.15.0, sentence-transformers 5.7.0, sae-lens 6.49.1 (installed, verified unusable for mT5), scikit-learn 1.9.0, sacrebleu 2.6.0, on a single NVIDIA A100-SXM4-40GB (driver 590.48.01, CUDA 13.0). Full pin set in `requirements-lock.txt`; environment snapshot in `results/env_report.json`; central configuration in `configs/experiment.yaml`.

| Stage | Commit | Script | Artifacts |
|---|---|---|---|
| 0 environment | `d15c8ae` | `scripts/00_check_environment.py` | `configs/experiment.yaml`, `results/env_report.json` |
| 1 corpus inventory | `b53c3af` | `scripts/01_inventory_corpus.py` | `results/corpus_inventory.csv` |
| 2 splits | `06e75a1` | `scripts/02_make_splits.py` | `data/splits/` |
| 3 fine-tuning | `1796887` | `scripts/03_train_mt5_detox.py` | `results/training/` |
| 4 generation | `5e8d74e` | `scripts/04_generate_baseline.py` | `results/generations/`, `results/baseline_generations.csv` |
| 5 evaluation | `185fb0b` | `scripts/05_evaluate_detox.py` | `results/evaluation/`, `results/baseline_metrics.csv` |
| 6 activations | `bb38cbd` | `scripts/06_collect_activations.py` | `results/activations/` (manifest + index tracked; tensors gitignored) |
| 7 probes | `0cf4163` | `scripts/07_train_probes.py` | `results/probes/` |
| 8 SAEs | `32c4679` | `scripts/08_train_sae.py` | `results/sae/` |
| 9 SAE features | `fcd64ae` | `scripts/09_analyze_sae_features.py` | `results/sae_features/` |
| 10 attribution | `8a0faa3` | `scripts/10_component_attribution.py` | `results/attribution/` |
| 11 ablation | `acfda4d` | `scripts/11_ablate_components.py` | `results/ablation/` (incl. `frozen_design.json`, `TEST_EVALUATED.txt`) |
| 12 steering vectors | `bce4563` | `scripts/12_build_steering_vectors.py` | `results/steering_vectors/` |
| 13 steering | `a16e48a`, `ca93d62`, `e3a6a96` | `scripts/13_generate_steered.py` | `results/steering/`, `results/steering_{dev,test}.csv` |
| 14 transfer | `a58302e` | `scripts/14_cross_lingual_transfer.py` | `results/transfer/` |
| 15 statistics | `b368710` | `scripts/15_bootstrap_results.py` | `results/statistics/` |
| 16–17 tables & figures | `8b9b617` | `scripts/16_make_paper_tables.py`, `scripts/17_make_paper_figures.py` | `results/tables/` (13 CSVs), `results/figures/` (8 PNG + 8 PDF) |

Determinism: seed 42 for splits and bootstrap; model seeds 42 / 1337 / 2024; beam-4 deterministic decoding, verified to reproduce byte-identically. Test-once discipline is enforced by digest-verified frozen designs plus `TEST_EVALUATED.txt` stamp files in `results/ablation/` and `results/steering/`.

Model checkpoints (3 × 2.2 GB), activation tensors (2.77 GiB) and SAE weights are gitignored as reproducible artifacts; their manifests and checksums are committed. `/workspace` on the run host was not volume-backed, so all durable state is in git.

**Six defects were found and corrected during the study**, each caught by a check rather than by inspection of the numbers: float16 overflow silently turning mT5 activations into `inf` (Stage 6); pooled normalisation statistics saved for only one pooling variant (Stage 6/7); a control baseline pinned to exactly 0.500 by construction (Stage 7); attribution columns misaligned with component names by hook-dict ordering (Stage 10); a pairing key that silently dropped an entire stage from the aggregation (Stage 15); and error bars showing between-language heterogeneity as though it were uncertainty (Stage 17). Each is documented in the corresponding commit message.

---

## 17. Conclusion

We set out to find cross-lingual detoxification circuits in mT5 and to test whether they transfer to Yoruba and isiXhosa. What we found is narrower than the question.

There is one robust causal result: a small set of decoder cross-attention heads, identified by attribution against a probe direction, measurably changes the toxicity of generated text when mean-ablated, and matched random components drawn from the same layers do not. Across three seeds, on a test split touched exactly once, the interval excludes zero. That is real, and the control structure supports it.

It is also modest and qualified. The effect size is small (d_z 0.236). The same intervention significantly degrades content preservation, so the joint score barely moves — the heads are better described as carrying toxic content forward than as implementing detoxification. The effect is uneven across languages, absent in Hindi and Ukrainian, and moves Yoruba and isiXhosa in opposite directions on output diversity. And the interpretability route that was supposed to explain *what* those heads compute failed: no sparse-autoencoder feature replicated across seeds, and the direction that best predicts the condition is distributed rather than localised in any latent we could find.

The steering and transfer results are null. We could not improve detoxification by adding a learned direction to the residual stream at the strengths we pre-specified, and vectors built from higher-resource languages did no better for Yoruba or isiXhosa than an arbitrary vector of the same length. We report these as nulls conditional on the intervention magnitude tested, not as evidence that steering cannot work.

For the two African languages the honest summary is more limited still. We have **no validated toxicity measurement for either**, and no native speaker examined a single output. What we can say is descriptive and worrying: the model collapses isiXhosa onto a handful of generic strings and repeats itself substantially in Yoruba, both languages carry heavy tokenisation costs, and their reference corpora encode a different annotation convention from the comparison set. Any future work should treat native-speaker evaluation and a validated toxicity classifier for these languages as prerequisites rather than refinements — without them, the central question for Yoruba and isiXhosa cannot actually be answered, only approximated.

What this study establishes is a method and a floor: a fully controlled pipeline in which associative, negative, causal and null evidence are separated and labelled, interventions are checked against matched random controls, coverage gaps are declared rather than papered over, and a result is called significant only when its interval says so. On that basis, the defensible claim is a single causal pathway with a harmful trade-off — not a detoxification circuit, and not a cross-lingual one.
