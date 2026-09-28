# Tracing and Steering Multilingual Text Detoxification Circuits in Large Language Models

## Abstract

Text detoxification models must remove harmful language without erasing meaning,
degrading fluency, or collapsing into generic outputs. These requirements become harder
in multilingual settings, where the internal representations supporting rewriting may be
partly shared across languages but unevenly transferable. We study toxic-to-detox
representations and activation steering across a nine-language corpus using mT5, mT0,
Llama-3-8B, Qwen2.5-7B-Instruct, and Aya-23-8B.

Our main tracing results come from multi-seed probes on Aya and Qwen. At layer 20,
Aya reaches mean pooled toxic-versus-detox AUC 0.8382 across two seeds and Qwen reaches
0.8427 across three seeds. Transfer from seven higher-resource languages to isiXhosa
and Yoruba is consistently lower: 0.5966 for Aya and 0.6054 for Qwen. This gap shows
that a representation signal shared across the corpus does not transfer uniformly to
the two lower-resource targets. We also analyze sparse autoencoder candidate features
and recovered activation-steering sweeps. In an exploratory 50-example Amharic mT5
sweep, steering strength produces a non-monotonic reference-alignment profile while
all outputs remain non-empty and unique.

We separate three questions often conflated in mechanistic detoxification work:
representation decodability, intervention-induced output change, and validated
detoxification improvement. Our final evaluation combines automatic preservation and
output-quality diagnostics with blinded human assessment of non-toxicity, meaning
preservation, fluency, and acceptability. [HUMAN-EVALUATION RESULTS TO INSERT.]

## 1 Introduction

Text detoxification asks a model to rewrite harmful or insulting content into a
non-toxic form while preserving its intended meaning and language. The task is already
difficult in English. It is more difficult across languages with different scripts,
tokenization behavior, data availability, and culturally specific forms of harmful
speech. A model may represent a toxic-to-detox transformation in a way that is
readily separable in one language while failing to transfer to another.

Mechanistic interpretability offers tools for examining this problem. Linear probes
can test whether hidden states distinguish toxic inputs from detoxified targets.
Sparse autoencoders can identify candidate latent features associated with the
transformation. Activation steering can test whether changing an internal
representation changes generated outputs. These tools answer different questions.
A high probe score shows decodability, not causality. An output change under steering
shows an intervention effect, not necessarily an improvement in non-toxicity or
utility. A complete multilingual analysis must keep these distinctions explicit.

We study three research questions.

**RQ1: Circuit structure.** Do multilingual models use shared or language-specific
internal features when transforming toxic text into detoxified text?

**RQ2: Cross-lingual transfer.** Do directions identified in higher-resource languages
transfer to Yoruba and isiXhosa, and how does this compare with within-language
directions?

**RQ3: Causal steering.** Can activation steering improve detoxification while
preserving semantic content, fluency, and output diversity?

Our results show that toxic-input and detox-target states are consistently decodable
in Aya and Qwen across fine-tuning seeds. Yet the signal transfers less strongly from
Amharic, Arabic, German, English, Spanish, Hindi, and Ukrainian to Yoruba and
isiXhosa. We find exploratory evidence that steering strength changes generated
outputs in a non-monotonic way. We use this evidence to motivate a strict
train--development--test confirmation protocol and a blinded human evaluation.

Our contributions are:

1. We provide multi-seed evidence that toxic-to-detox representations are linearly
   decodable in Aya-23-8B and Qwen2.5-7B-Instruct, with a consistent drop in
   high-resource-to-low-resource transfer.
2. We trace layerwise transfer behavior in Qwen and analyze seed-specific SAE
   candidate features across model families.
3. We document strength-dependent steering behavior in recovered mT5/MT0 artifacts
   and define a paired, controlled confirmation protocol for multilingual steering.
4. We separate representation, causal, and behavioral claims through an explicit
   evidence hierarchy, avoiding the use of preservation-only composite scores as
   toxicity or safety outcomes.

## 2 Task, Data, and Models

### 2.1 Multilingual detoxification task

Each example contains a toxic input \(x\), a human detoxified target \(y\), and a
language label \(l\). A model receives \(x\) and generates a rewrite
\(\hat{y}\). The intended rewrite reduces harmful or insulting language while
preserving the underlying proposition, stance, and target language.

The audited expanded corpus contains nine languages: Amharic (`am`), Arabic (`ar`),
German (`de`), English (`en`), Spanish (`es`), Hindi (`hi`), Ukrainian (`uk`),
isiXhosa (`xh`), and Yoruba (`yo`). For Qwen, the preserved training manifests record
2,583 training examples, 558 development examples, and 553 frozen test examples.
The test split hash is
`d31b7c18abc9be003d916d07037b868fa32f0d013298e735fc0e879f5f4e0d61`.

### 2.2 Models

We analyze five model families: mT5, mT0, Llama-3-8B-Instruct, Qwen2.5-7B-Instruct,
and Aya-23-8B. The primary multi-seed tracing analyses use Aya adapters trained with
seeds 42 and 1337 and Qwen adapters trained with seeds 42, 1337, and 2025. We use
mT5 and mT0 as encoder--decoder analyses and exploratory steering cases, and Llama
as a decoder-only comparison in supplementary analyses.

### 2.3 Evaluation principles

We distinguish four evaluation categories.

1. **Representation decodability:** a linear classifier distinguishes hidden states
   associated with toxic inputs and detoxified targets.
2. **Output intervention:** a steering direction changes the generated output.
3. **Preservation and output quality:** semantic similarity, reference overlap,
   copy rate, emptiness, length, and output diversity diagnose generation behavior.
4. **Human detoxification quality:** blinded annotators assess non-toxicity, meaning
   preservation, fluency, and overall acceptability.

The legacy repository evaluator defines
\[
J_{\mathrm{legacy}} = \frac{1 + \mathrm{SIM} + \mathrm{chrF}}{3},
\]
because its STA component is fixed to 1. We therefore do not interpret this composite
as an independent toxicity, safety, or detoxification measure. We report it only when
needed for compatibility with earlier artifacts, and we report its components
separately.

## 3 Tracing Toxic-to-Detox Representations

### 3.1 Probe setup

For Aya and Qwen, we extract hidden states for toxic inputs and their paired detoxified
targets. Linear logistic-regression probes predict whether a representation comes from
a toxic input or a detoxified target. The pooled and higher-resource scores use the
stored random-example split protocol with fixed random state 42 and stratification by
the binary toxic-versus-detox label.

The higher-resource set comprises Amharic, Arabic, German, English, Spanish, Hindi,
and Ukrainian. The transfer score trains on all representations from these seven
languages and evaluates on isiXhosa and Yoruba. This is a specific
higher-resource-to-lower-resource transfer condition, not leave-one-language-out
evaluation. Because the random split is not grouped by original toxic/detox pair,
paired states may occur on opposite sides of a split; we treat the result as
representation decodability under the stored protocol.

### 3.2 Multi-seed probe results

Table 1 reports layer-20 probe performance. Aya reaches pooled AUC
\(0.8382 \pm 0.0012\) across two seeds, and Qwen reaches
\(0.8427 \pm 0.0175\) across three seeds. The corresponding
higher-resource-to-isiXhosa/Yoruba AUCs are lower:
\(0.5966 \pm 0.0014\) for Aya and \(0.6054 \pm 0.0219\) for Qwen.

[INSERT TABLE 1: MULTI-SEED LAYER-20 PROBE RESULTS]

The approximately 0.24-AUC reduction appears in both model families. This result
supports a partially shared representation signal: a probe trained in one language
group remains above chance on Yoruba and isiXhosa, but transfers much less effectively
than it separates states under the pooled protocol.

[INSERT FIGURE 1: POOLED VS HR-TO-LR PROBE AUC]

### 3.3 Qwen layer profile

Qwen provides a three-seed layer profile at layers 10, 15, and 20. Pooled
random-example AUC generally rises toward later layers. Higher-resource-to-lower-resource
transfer remains lower at every measured layer, but layer 20 produces the strongest
stored transfer score for all three seeds: 0.5975, 0.6302, and 0.5884. Layer 20 is
not the pooled-AUC optimum in every seed; for seed 1337, layer 15 slightly exceeds it.
We therefore describe layer 20 as the most consistently strong stored transfer layer,
not as a universally optimal layer.

[INSERT FIGURE 2: QWEN LAYER PROFILE]

### 3.4 SAE candidate features

We train seed-specific sparse autoencoders on layer-20 toxic and detox activations and
rank latent dimensions by their mean activation difference between detoxified targets
and toxic inputs. For Aya seed 42, the SAE has 8,192 hidden dimensions and is trained
on 4,000 activations from 2,000 toxic/detox pairs. Qwen has corresponding SAE summaries
for three seeds.

These ranked dimensions are candidate features, not validated semantic concepts or
causal circuits. Independently trained SAE dictionaries have permutation ambiguity, so
raw latent indices cannot establish cross-seed feature correspondence. We use the SAE
analysis to identify candidate directions and to motivate intervention experiments;
full feature inventories appear in the appendix.

## 4 Cross-Lingual Transfer

### 4.1 Shared signal and transfer gap

The probe results provide the first transfer test. A classifier trained on the seven
higher-resource languages remains above chance on Yoruba and isiXhosa, but its
performance is much lower than pooled performance. This pattern is consistent with
a representation that contains both shared and language-sensitive components.

The observed gap does not imply that low-resource languages lack relevant
representations. It may arise from lexical and morphological differences, tokenization,
training-data imbalance, source-domain differences, or a mismatch between the
representations captured by the selected layer and the transformation mechanism in
Yoruba and isiXhosa.

### 4.2 Exploratory cross-language steering

Recovered cross-language steering artifacts include language-specific directions at
encoder layer 6 and English-to-Yoruba/isiXhosa output comparisons. In the archived
copy-rate table, English-transfer steering produces zero exact copies for both Yoruba
and isiXhosa, while within-language copy rates are 0.0333 for Yoruba and 0.0000 for
isiXhosa. These values show that cross-language directions alter generation behavior,
but copy rate alone cannot determine whether the outputs are less toxic, more
meaning-preserving, or more acceptable.

Accordingly, we treat these artifacts as exploratory evidence of cross-language output
sensitivity. Our final confirmation protocol computes directions on training data,
selects layers and strengths on development data, and compares within-language,
English-transfer, random-direction, and shuffled-pair directions against paired
no-intervention outputs on the frozen test set.

[INSERT TABLE 2: CLEAN CROSS-LINGUAL CONFIRMATION RESULTS WHEN AVAILABLE]

## 5 Steering Toxic-to-Detox Directions

### 5.1 Exploratory Amharic strength sweep

A recovered mT5 sweep applies an encoder-layer-6 steering direction to the same 50
Amharic examples at strengths 0.5, 1.0, 2.0, 3.0, and 5.0. All strengths produce
50 unique and non-empty outputs. Reference overlap with the human detoxified target
peaks at strength 3.0 (chrF 0.4292), compared with 0.3916 at strength 0.5. At strength
5.0, reference overlap falls to 0.4154 while edit similarity to the toxic input rises.
This non-monotonic pattern is consistent with a strength--utility trade-off.

The sweep is exploratory: it has no strength-0 baseline, uses a test-derived direction,
contains no independent toxicity measure, and is not replicated across seeds. We use
it to motivate—not to replace—the controlled confirmation experiment.

[INSERT FIGURE 3 OR APPENDIX FIGURE A1: AMHARIC STRENGTH SWEEP]

### 5.2 Controlled steering confirmation

[TO BE COMPLETED AFTER CLEAN RERUN.]

For the final intervention analysis, we estimate directions from training data only.
We select layer and strength on a stratified development subset and retain the complete
search record. On the frozen test split, we generate paired outputs under: no
intervention; within-language direction; English-transfer direction; matched-norm
random direction; and shuffled-pair direction. We report per-language paired bootstrap
confidence intervals for non-toxicity, meaning preservation, fluency, acceptability,
semantic similarity, reference overlap, copying, emptiness, length, and output
diversity.

### 5.3 Human evaluation

[TO BE COMPLETED WHEN ANNOTATIONS RETURN.]

We prepared 500 sampled outputs across nine languages and four model families for
blinded human evaluation. The protocol assigns at least two annotators per output and
collects binary non-toxicity and acceptability judgments plus 1--5 meaning-preservation
and fluency ratings. We will report annotator agreement, per-language sample counts,
and confidence intervals with the final labels.

## 6 Discussion

The combined evidence supports a careful version of the circuit hypothesis. Toxic and
detoxified text occupy linearly separable regions of representation space in two
multilingual decoder models across multiple fine-tuning seeds. The transfer gap shows
that this representation is not uniformly language-invariant. The exploratory steering
results show that interventions can change outputs in a strength-sensitive way without
obvious output collapse. They do not, by themselves, prove that a shared causal
detoxification circuit has been isolated.

This distinction matters for multilingual safety research. Automatic preservation
metrics can quantify whether a rewrite remains close to its input or reference, but
they cannot substitute for a valid non-toxicity measure. Likewise, a probe can reveal
a useful representational signal without establishing that the signal is sufficient
for controllable generation. The controlled intervention and human-evaluation stages
are therefore necessary for interpreting steering as detoxification rather than as
generic output perturbation.

## 7 Conclusion

We trace toxic-to-detox representations in multilingual rewriting models and show
multi-seed decodability in Aya and Qwen. Transfer from seven higher-resource languages
to Yoruba and isiXhosa is weaker but remains above chance under the stored probe
protocol. Recovered steering sweeps show strength-dependent generation behavior and
motivate a controlled confirmation design. Together, these results support a view of
multilingual detoxification as partially shared but language-sensitive, and they show
why tracing, steering, and behavioral validation must be evaluated separately.

## Limitations

Our primary probe scores use random-example splits that are not grouped by original
toxic/detox pair or language. The reported higher-resource-to-lower-resource transfer
condition evaluates only Yoruba and isiXhosa and is not leave-one-language-out. Some
recovered intervention artifacts use test-derived directions, lack paired
strength-zero baselines, or record only a limited subset of output metrics; we label
these analyses exploratory and do not use them to claim validated detoxification
improvement. The current human-evaluation sample has been sent for annotation, but
completed labels, agreement statistics, and human outcome estimates are not yet
available in this draft. The dataset, language coverage, and automatic metrics may
not capture all culturally specific forms of harmful language or all legitimate
rewriting choices.

## Ethical Considerations

Text detoxification can suppress abusive language, but it can also erase justified
anger, political critique, identity-related expression, or culturally specific
rhetoric. We therefore distinguish harmful insult reduction from disagreement or
stance removal and require human review for claims about non-toxicity and
acceptability. The corpus includes sensitive and offensive text. We limit quotation
of examples in the paper, protect annotator well-being through clear instructions and
the option to skip items, and avoid presenting automatic proxy scores as safety
outcomes. Results should not be used to censor speakers or to treat low-resource
language varieties as inherently more toxic or less fluent.
