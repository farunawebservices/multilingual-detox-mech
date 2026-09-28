# Draft Results: Probe Decodability and Transfer

## Toxic-versus-detox representations are decodable under the stored probe protocol

We evaluated linear probes that distinguish representations of toxic inputs from
representations of paired detoxified targets. For Aya-23-8B, layer-20 probes achieved
pooled random-example-split AUCs of 0.8374 and 0.8391 across seeds 42 and 1337,
respectively. For Qwen2.5-7B-Instruct, layer-20 pooled AUC ranged from 0.8274 to
0.8617 across seeds 42, 1337, and 2025. Thus, under the stored random-example-split
protocol, toxic-input and detoxified-target states are linearly separable in both
model families.

This finding is evidence of representation decodability, not an independent measure
of toxicity, safety, detoxification quality, or causal control. In particular, pooled
random splits are not grouped by original pair or language, so paired toxic and
detoxified examples may occur on opposite sides of a split.

## Transfer to isiXhosa and Yoruba is weaker

We next trained probes on representations from Amharic, Arabic, German, English,
Spanish, Hindi, and Ukrainian, and evaluated them on isiXhosa and Yoruba. Aya's
layer-20 high-resource-to-low-resource AUC was 0.5956 and 0.5977 across its two
seeds (mean 0.5966, SD 0.0014), compared with mean pooled AUC 0.8382 (SD 0.0012).
Qwen's corresponding transfer AUC was 0.5975, 0.6302, and 0.5884 across three seeds
(mean 0.6054, SD 0.0219), compared with mean pooled AUC 0.8427 (SD 0.0175).

The transfer scores are lower by approximately 0.24 AUC for both model families.
This indicates weaker high-resource-to-low-resource transfer under the evaluated
protocol. It does not establish a general failure of cross-lingual generalization,
because only two evaluation languages are in the low-resource test group and no
leave-one-language-out evaluation was performed.

## Layer profile in Qwen

For Qwen, pooled random-example AUC generally increased from layer 10 to later
layers. High-resource-to-low-resource transfer was lower at every stored layer, but
layer 20 had the strongest stored transfer AUC in all three seeds: 0.5975, 0.6302,
and 0.5884. Layer 20 was not the highest pooled-AUC layer in every seed; for seed
1337, layer 15 slightly exceeded layer 20 on pooled AUC. We therefore describe layer
20 as the most consistently strong stored transfer layer, rather than as universally
optimal.
