# Stage 13 Completion Checklist

## ✅ Completed
- [x] Cross-model comparison table (mT5, mT0, Llama-3-8B)
- [x] Per-language metrics tables
- [x] Mechanistic comparison table
- [x] Dataset statistics table
- [x] Figure 1: Cross-model performance (4-panel)
- [x] Figure 2: Per-language metrics (mT5 vs mT0)
- [x] Figure 3: Copy rate vs SIM scatter
- [x] Figure 4: Mechanistic results (probe + ablation)
- [x] All evaluation data saved (predictions, metrics)
- [x] All training artifacts saved (checkpoints, logs)
- [x] Environment pinned (requirements-lock.txt)
- [x] Git repository up to date

## ⏳ Remaining Before Paper Submission
- [ ] Human evaluation (Stage 9)
  - [ ] Sampling protocol
  - [ ] Annotation guidelines
  - [ ] Annotator recruitment
  - [ ] Data collection
  - [ ] Agreement analysis
- [ ] Advanced mechanistic (Stage 10-12)
  - [ ] SAE training
  - [ ] Activation steering
  - [ ] Cross-lingual transfer baseline
- [ ] Paper writing
  - [ ] Abstract
  - [ ] Introduction
  - [ ] Methods
  - [ ] Results
  - [ ] Discussion
  - [ ] References
- [ ] Supplementary materials
  - [ ] Extended tables
  - [ ] Additional examples
  - [ ] Code repository link

## 📊 Key Numbers for Paper
- Corpus: 553 toxic-detox pairs (9 languages)
- Models: 4 main (mT5-175, mT5-400+, mT0-400+, Llama-3-8B)
- Seeds: 3 per model (42, 1337, 2024)
- Best copy rate: 2.2% (Llama-3-8B)
- Best J score: 0.85 (mT5 historical)
- Probe AUC: 0.88-0.99 (all models)
- Cross-lingual transfer: 2-4% copy rate in yo/xh

## 🎯 RQ Answers
- RQ1: ✅ Shared circuits exist (AUC 0.88-0.99)
- RQ2: ✅ Cross-lingual transfer works (yo/xh copy 2-4%)
- RQ3: ✅ Causal intervention reduces copy (Δ -4% to -9%)

---
*Last updated: 2026-09-23*
