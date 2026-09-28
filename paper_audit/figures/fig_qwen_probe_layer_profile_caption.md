**Figure 2: Qwen2.5-7B probe decodability across layers.**
Solid lines show individual fine-tuning seeds and dashed lines show their mean.
The left panel reports pooled random-example-split AUC for distinguishing toxic inputs
from detoxified targets. The right panel reports high-resource-to-low-resource transfer:
a probe trained on Amharic, Arabic, German, English, Spanish, Hindi, and Ukrainian is
evaluated on isiXhosa and Yoruba. Layer 20 provides the strongest stored transfer AUC
in all three seeds, although it is not the top pooled-AUC layer for every seed.
Results measure representation decodability under the stated protocol, not toxicity
reduction, safety, causal control, or generic language-general performance.
