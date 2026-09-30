# Aya-23-8B Confirmatory Steering Protocol

Base model: CohereLabs/aya-23-8B
Primary LoRA adapter seed: 42
Replication adapter seed: 1337, conditional on completion of the primary run

Data root: results/training_v2_400plus/seed_2024
Direction-estimation split: train.csv only
Layer/strength-selection split: dev.csv only
Final evaluation split: test.csv only

Direction source language: en
Target languages: xh, yo
Within-language comparison directions: xh and yo

Candidate layers: 16, 20
Strength grid: 0.0, 0.25, 0.5, 1.0, 2.0
Decoding: deterministic greedy decoding; do_sample=False
Maximum new tokens: 128

Frozen-test conditions:
C0 baseline
C1 English-transfer direction
C2 within-language direction
C3 norm-matched random-direction control
C4 shuffled-pair direction control

Primary outcomes:
- blinded human non-toxicity
- blinded human meaning preservation

Secondary outcomes:
- fluency
- acceptability
- chrF to detox reference
- LaBSE similarity to detox reference
- LaBSE similarity to toxic input
- toxic-input copy rate
- reference-copy rate
- empty-output rate
- output length
- output diversity

No frozen-test output will be inspected to select a layer, strength, source direction,
or decoding configuration.
