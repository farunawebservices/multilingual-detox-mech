# Draft Methods and Limitations

## Task and data provenance

We study representations associated with paired toxic-input and detoxified-target text
in a nine-language multilingual rewriting corpus. The audited Qwen training manifests
record common frozen splits of 2,583 training examples, 558 development examples, and
553 test examples across seeds 42, 1337, and 2025. The shared test split has SHA-256
`d31b7c18abc9be003d916d07037b868fa32f0d013298e735fc0e879f5f4e0d61`.

Our primary representation analyses concern Aya-23-8B and Qwen2.5-7B-Instruct.
For both model families, probes distinguish hidden states computed from toxic inputs
from hidden states computed from their paired detoxified targets. We report the stored
layer-20 results because Aya was evaluated at layer 20 and Qwen layer 20 produced the
strongest stored high-resource-to-low-resource transfer score in each available seed.

## Probe protocols

For pooled and high-resource analyses, the stored Aya and Qwen scripts use a
random-example split stratified by the binary toxic-versus-detox label, with
`random_state=42`. The high-resource subset comprises Amharic, Arabic, German,
English, Spanish, Hindi, and Ukrainian. For the transfer analysis, a logistic
regression probe is trained on all examples from these seven languages and evaluated
on isiXhosa and Yoruba.

We describe these measurements as representation decodability under the stored
protocol. Random-example splits are not grouped by original pair or language, so a
toxic input and its paired detoxified target may occur on opposite sides of a random
split. The high-resource-to-low-resource analysis is not leave-one-language-out
evaluation and does not assess every language as a held-out target.

## Behavioral metrics

The recovered evaluator scripts for MT5, MT0, Llama, Qwen, and Aya calculate semantic
similarity as LaBSE cosine similarity between the toxic input and model prediction and
calculate chrF against the detoxified reference. They compute a copy rate through
normalized exact matching. However, they set STA to the constant value 1.0 and define
legacy J as `(STA + SIM + chrF) / 3`. Accordingly, we do not interpret legacy J or
STA as toxicity, safety, or detoxification measures. We refer to chrF as a
reference-overlap proxy rather than fluency.

## Intervention audit

We audit rather than aggregate existing intervention outputs. Aya's retained layer-20
steering run selected strength 0.0 on development data; the corresponding 553-example
baseline and nominally steered test files are byte-identical, yielding a verified null
control. The retained Aya layer-24 sweep lacks the full development sweep record and
paired test predictions needed for confirmatory inference.

We exclude Qwen steering effect estimates because its direction was constructed from
test examples, only selected-strength test predictions were retained, and its stored
delta subtracts a development baseline from a test output. We similarly exclude Llama
steering because its stored values are explicitly estimated ablation proxies. MT5 and
MT0 ablation runs are retained only as exploratory random-head sensitivity analyses:
their scripts choose ablated head orders using random permutations rather than a
targeted mechanistic selection procedure.

## Scope

Our claims are limited to audited representation decodability and the observed
high-resource-to-low-resource transfer gap. They do not establish causal
controllability, a universal detoxification circuit, independent multilingual toxicity
measurement, or behavioral safety improvement.
