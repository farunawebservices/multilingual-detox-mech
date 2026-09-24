# Qwen2.5-7B-Instruct Multilingual Detoxification: Recovered Experiment Report

**Status:** Behavioral QLoRA results are reconstructed from saved notebook console output. Raw adapter checkpoints, prediction CSVs, activations, and probe artifacts were stored only in Colab's ephemeral `/content` filesystem and were lost after a runtime reset. Consequently, this report presents only values explicitly recorded in the available logs. Metrics whose raw inputs were not retained—such as semantic similarity (SIM), fluency (FL), toxicity metrics (STA and \(\Delta\mathrm{STA}\)), and composite quality \(J\)—are intentionally reported as unavailable rather than estimated.

## 1. Scope and dataset

### Model and training objective

The experiment fine-tuned `Qwen/Qwen2.5-7B-Instruct` for multilingual toxic-to-detox text rewriting. The instruction format required the model to rewrite toxic text into non-toxic text while preserving original meaning and language, returning only the rewrite.

Training used 4-bit QLoRA on an NVIDIA A100-SXM4-40GB GPU:

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | bfloat16 |
| LoRA rank \(r\) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 10,092,544 |
| Total parameters | 7,625,709,056 |
| Trainable share | 0.1323% |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| Learning rate | \(2\times10^{-4}\) |
| Optimizer | `paged_adamw_8bit` |
| Decoding | Greedy / deterministic generation |
| Maximum input length | 512 tokens |
| Maximum new tokens | 128 |

For seeds 42, 1337, and 2025, checkpoints were selected by minimum aggregate development loss. The earlier seed-2024 run was a preliminary run that evaluated against the test split during training and is excluded from the main three-seed summary.

### Frozen corpus and splits

The frozen corpus contained 3,694 toxic-to-detox pairs in nine languages.

| Language | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Amharic (`am`) | 280 | 60 | 60 | 400 |
| Arabic (`ar`) | 280 | 60 | 60 | 400 |
| German (`de`) | 280 | 60 | 60 | 400 |
| English (`en`) | 280 | 60 | 60 | 400 |
| Spanish (`es`) | 280 | 60 | 60 | 400 |
| Hindi (`hi`) | 280 | 60 | 60 | 400 |
| Ukrainian (`uk`) | 280 | 60 | 60 | 400 |
| isiXhosa (`xh`) | 317 | 68 | 68 | 453 |
| Yoruba (`yo`) | 306 | 70 | 65 | 441 |
| **All languages** | **2,583** | **558** | **553** | **3,694** |

## 2. Training stability

All three primary QLoRA runs completed three epochs, produced finite development losses, saved a best adapter by development loss, and generated non-empty outputs on all 553 test examples.

| Seed | Dev loss, epoch 1 | Dev loss, epoch 2 | Dev loss, epoch 3 | Best recorded dev loss | Selected epoch |
|---:|---:|---:|---:|---:|---:|
| 42 | 1.030322 | 1.004706 | 1.044126 | **1.004706** | 2 |
| 1337 | 1.037371 | 1.001582 | 1.047799 | **1.001582** | 2 |
| 2025 | 1.030547 | 0.994647 | 1.040961 | **0.994647** | 2 |

All seeds reached their best recorded development loss in epoch 2, followed by a higher epoch-3 loss. This consistent trajectory supports selecting the epoch-2 checkpoint and suggests that further fixed-epoch training would risk mild overfitting.

## 3. Test diagnostics

### Aggregate diagnostics

| Seed | Test examples | Empty-output rate | Exact-copy rate | Mean output characters |
|---:|---:|---:|---:|---:|
| 42 | 553 | 0.0000 | 0.0072 | 58.00 |
| 1337 | 553 | 0.0000 | 0.0398 | 60.36 |
| 2025 | 553 | 0.0000 | 0.0380 | 60.34 |
| **Mean across seeds** | **553** | **0.0000** | **0.0283** | **59.57** |

The model produced no empty outputs in any primary seed. Exact copying was low overall but varied across initialization, particularly in Amharic.

### Per-language exact-copy rate

Exact-copy rate is the proportion of generated strings identical to the toxic source string after surrounding-whitespace normalization. It is a collapse diagnostic, not a direct measure of toxicity or semantic preservation.

| Language | Test \(n\) | Seed 42 | Seed 1337 | Seed 2025 | Mean |
|---|---:|---:|---:|---:|---:|
| Amharic (`am`) | 60 | 0.0500 | 0.2667 | 0.2000 | **0.1722** |
| Arabic (`ar`) | 60 | 0.0000 | 0.0167 | 0.0833 | **0.0333** |
| German (`de`) | 60 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| English (`en`) | 60 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| Spanish (`es`) | 60 | 0.0000 | 0.0333 | 0.0000 | **0.0111** |
| Hindi (`hi`) | 60 | 0.0000 | 0.0000 | 0.0167 | **0.0056** |
| Ukrainian (`uk`) | 60 | 0.0000 | 0.0000 | 0.0167 | **0.0056** |
| isiXhosa (`xh`) | 68 | 0.0147 | 0.0294 | 0.0294 | **0.0245** |
| Yoruba (`yo`) | 65 | 0.0000 | 0.0154 | 0.0000 | **0.0051** |
| **Macro mean** | — | **0.0072** | **0.0401** | **0.0385** | **0.0286** |

### Per-language mean output length

| Language | Seed 42 | Seed 1337 | Seed 2025 | Mean |
|---|---:|---:|---:|---:|
| Amharic (`am`) | 62.78 | 64.78 | 64.85 | 64.14 |
| Arabic (`ar`) | 55.65 | 57.18 | 55.60 | 56.14 |
| German (`de`) | 100.53 | 99.65 | 100.97 | 100.38 |
| English (`en`) | 47.78 | 48.02 | 48.32 | 48.04 |
| Spanish (`es`) | 60.63 | 66.60 | 65.37 | 64.20 |
| Hindi (`hi`) | 52.07 | 53.25 | 50.97 | 52.09 |
| Ukrainian (`uk`) | 46.95 | 48.25 | 47.22 | 47.47 |
| isiXhosa (`xh`) | 54.12 | 56.96 | 62.16 | 57.75 |
| Yoruba (`yo`) | 43.26 | 49.91 | 48.38 | 47.18 |

## 4. Metrics availability

The original planned evaluation includes STA, \(\Delta\mathrm{STA}\), SIM, FL, \(J\), output diversity, and collapse diagnostics. Only empty-output rate, exact-copy rate, and output length were explicitly printed and recoverable after the Colab runtime reset.

| Metric | Definition / intended role | Status | Reason |
|---|---|---|---|
| STA | Toxicity / non-toxicity classifier score | **Not available** | Per-example predictions and classifier outputs were lost |
| \(\Delta\mathrm{STA}\) | Change in toxicity relative to input | **Not available** | Per-example predictions and classifier outputs were lost |
| SIM | Semantic similarity / meaning preservation | **Not available** | Per-example predictions and embeddings or metric outputs were lost |
| FL | Fluency score | **Not available** | Per-example predictions and scoring artifacts were lost |
| \(J\) | Predefined composite automatic score | **Not available** | Requires STA, SIM, FL, and raw predictions |
| Empty-output rate | Fraction of blank generations | **Available** | Printed in seed diagnostics |
| Exact-copy rate | Fraction of outputs identical to toxic sources | **Available** | Printed in seed diagnostics |
| Unique-output rate | Output diversity / non-collapse | **Not available** | Requires raw prediction CSVs |
| Top-output share | Frequency of the most common output | **Not available** | Requires raw prediction CSVs |

**Interpretation constraint:** It would be inaccurate to infer SIM, STA, FL, or \(J\) from copy rate, output length, references, or qualitative examples. Those metrics should be recomputed only after raw test predictions are regenerated and preserved.

## 5. Qualitative findings from logged test outputs

The following examples were printed by the evaluation scripts and are included as illustrative evidence only. They should not replace blinded human evaluation or complete automatic metrics.

### English: strong detoxification in common cases

| Toxic input | Reference detox | Seed-42 prediction |
|---|---|---|
| `And how can you justify a moronic 2 day block?` | `And how can you justify a 2 day block?` | `And how can you justify a 2 day block?` |
| `Bullshit William Penn used the phrase too.` | `William Penn used the phrase too.` | `William Penn used the phrase too.` |
| `Fuck you Roger Swivel, talk....` (dev) | `Dear Roger Swivel, let's talk.` | `Roger Swivel, talk.` |

The English examples show direct removal or softening of profanity and insults, often preserving the central proposition. However, the output can occasionally over-delete identifying or referential content.

### Yoruba: some successful detoxification, but generic reformulation

| Toxic input | Reference detox | Logged prediction |
|---|---|---|
| `@username awon omo ole` | `@username ẹ ń hùwà bí ẹni tí ń ji nnkan` | Seed 42: `@username ìwà rẹ̀ nílò ìtúnṣe díẹ̀` |
| `set awon omo buruku #unilag #madeinlagos` | `set awon omo #unilag #madeinlagos` | Seed 42: `Ìwà rẹ̀ nílò ìtúnṣe díẹ̀ #unilag #madeinlagos` |
| `set awon omo buruku #unilag #madeinlagos` | `set awon eniyan ti iwa won ko daa #unilag #madeinlagos` | Seed 2025: `awon ti ko ni iwa buruku #unilag #madeinlagos` |

Yoruba outputs often remove overtly insulting terms, but they may replace the source with generic phrases such as “conduct needs some improvement,” reducing lexical specificity and potentially weakening semantic preservation.

### isiXhosa: partial lexical edits but unstable detoxification

| Toxic input | Reference detox | Logged prediction |
|---|---|---|
| `yawthetha umsunu lo dzudzu yazi. URL` | `Uyathetha izinto ezingaqondakaliyo, yazi.` | Seed 42: `Yawthetha umsunu lo dzudzu yazi.` |
| `brother wam ... ndizovotela umnqundu` | `brother wam ... ndizovotela umntu` | Seed 42: `Brother wam ... ndizavotela umntu` |
| `ngumnqundu lo ref. ... ngumsunu qha ke!` | `Lo ref ubukeka enecala; wenza izinto ezikhetha umntu wakhe. Akulunganga oko.` | Seed 42: `Nceda lo ref. wenza incanca kanyoko wakhe ngok. Nceda qha ke!` |

The isiXhosa results include some useful substitutions, such as replacing `umnqundu` with `umntu`, but also retain toxic language and sometimes become incoherent. These examples require native-speaker assessment before claiming usable detoxification quality.

### Amharic: highest seed sensitivity and copy-collapse risk

| Toxic input | Reference detox | Logged prediction |
|---|---|---|
| `Egotistic ብሬንለስ ደደብ ብቻ ነው አገሩን ሚመራው` | `ለራሱ ብቻ የሚያስብ ነው አገሩን የሚመራው` | Seed 42: `በሰው የሚያጠፋ መልክ ብቻ ነው አገሩን ሚመራው` |
| same input | same reference | Seed 1337: `በሬንለስ ደደብ ብቻ ነው አገሩን ሚመራው` |
| same input | same reference | Seed 2025: `አገሩን ሚመራው አይደለም` |

Amharic had the highest exact-copy rates and visible qualitative variability across seeds. This is the clearest language-specific stability issue in the recovered diagnostics.

## 6. Representation analysis: completed coverage, restricted interpretation

Activation collection was completed for seeds 42, 1337, and 2025 over the entire **training split** and all nine languages. For each language and training example, hidden-state representations were extracted at 28 transformer layers with hidden dimension 3,584. The collected representations compared a pooled prompt-side span with a pooled target-side detox span.

Initial linear-probe outputs, printed before the runtime reset, reported:

- within-language toxic-versus-detox AUC of 1.000 at layer 1 for all nine languages;
- high-resource-to-target transfer AUC of 1.000 at layer 1 when probes trained on English, German, and Spanish were evaluated on Yoruba and isiXhosa.

These results demonstrate that the **specific extracted representation sets were linearly separable** under the initial pipeline. However, they do **not yet establish a shared detoxification circuit** because the pooled prompt-side and target-side spans differ systematically in position, prompt template context, lexical content, and causal-generation role. Such differences can yield trivial separability.

### Required controls before using probes for RQ1/RQ2 claims

1. Extract matched, content-controlled spans rather than pooling the entire prompt with the system/user template.
2. Use paired toxic/detox representations at comparable token positions where feasible.
3. Include a random-label control and a lexical/length-matched control.
4. Evaluate probe generalization on held-out development examples, not only the activation-construction split.
5. Quantify layerwise performance over all layers rather than choosing the first layer with maximal AUC.
6. Retain and version raw activation files, probe coefficients, split manifests, and all metric CSVs.

Accordingly, the representation collection is complete in dataset/language/seed coverage, but the initial numerical probe result should be treated as **preliminary pending controlled reanalysis**, not as final evidence for RQ1 or RQ2.

## 7. Steering analysis

The report intentionally does not include steering as a substantive result. A small early-layer sweep was attempted before artifact loss, but it is not promoted because the intervention was not yet selected through the preregistered dev-only procedure, did not include the full matched-random control suite, and its raw outputs were not retained.

The next valid steering phase should:

1. derive a candidate direction from training activations only;
2. choose layer and strength using the dev split only;
3. compare against norm-matched random directions;
4. freeze the configuration and hash it;
5. evaluate the test split once;
6. report toxicity, semantic preservation, fluency, diversity, collapse diagnostics, and paired bootstrap confidence intervals.

## 8. Conclusions supported by recovered evidence

1. Qwen2.5-7B-Instruct can be fine-tuned stably with 4-bit QLoRA on the frozen 3,694-pair multilingual corpus.
2. Across seeds 42, 1337, and 2025, all 553 test inputs received non-empty deterministic outputs.
3. Exact copying was generally low, with a mean overall copy rate of 2.83% across seeds; however, Amharic was notably more variable, averaging 17.22% exact copies.
4. Logged qualitative examples indicate meaningful detoxification in English and some successful edits in Yoruba and isiXhosa, alongside persistent residual toxicity, generic reformulation, incoherence, and language-specific instability.
5. The complete automatic evaluation suite—STA, \(\Delta\mathrm{STA}\), SIM, FL, \(J\), diversity/collapse metrics, and bootstrap intervals—cannot be reconstructed accurately without the lost prediction files and must be regenerated in a persistent-storage rerun.
6. The activation data were collected across all training examples, languages, and primary seeds, but initial probe outcomes require position-controlled and lexical controls before being interpreted as evidence of shared or transferable detoxification mechanisms.

## 9. Reproduction and preservation protocol

Future reproduction should use persistent storage from the first cell:

- Persist the repository and all result directories on Google Drive or a persistent-volume GPU service.
- After every completed seed, save: adapter files, trainer state, logs, test prediction CSV, diagnostics, model/config manifest, and checksums.
- Commit code, configuration files, small CSV/JSON/Markdown artifacts, and manifests to Git immediately after every stage.
- Store large adapters and activation tensors using a persistent-volume path or a dedicated artifact store; avoid committing multi-gigabyte binary tensors directly to standard Git.
- Generate all paper tables and figures from retained raw prediction/metric files rather than manually transcribed console logs.

## Appendix: recovered seed-level output-length diagnostics

The output-length values are reproduced above because they were explicitly emitted by the evaluation scripts. They provide only a weak diagnostic for truncation or generic shortening; they are not a measure of semantic preservation or fluency.
