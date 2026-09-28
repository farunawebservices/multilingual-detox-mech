# Required updates before ARR submission

## Human evaluation
- [ ] Replace the placeholder text in `sections/abstract.tex`.
- [ ] Add a human-evaluation results table with non-toxicity, meaning, fluency,
      acceptability, sample counts, and annotator agreement.
- [ ] State annotator count, language coverage, compensation/ethics details, and
      disagreement procedure.
- [ ] Verify that the anonymized labels and annotation instructions contain no
      identifying metadata.

## Clean steering confirmation
- [ ] Replace the result slot in `sections/steering.tex`.
- [ ] Add paired no-intervention, within-language, English-transfer, random-direction,
      and shuffled-pair direction results.
- [ ] Include train-only direction estimation and dev-only layer/strength selection.
- [ ] Add paired bootstrap confidence intervals.
- [ ] State which model/seed configurations replicate.

## Literature and references
- [ ] Replace all placeholder citations in `references.bib`.
- [ ] Add related work on multilingual detoxification, probing, SAEs, activation
      steering, multilingual transfer, LoRA/QLoRA, chrF, and LaBSE.
- [ ] Verify every citation and publication venue.

## ARR compliance
- [ ] Replace local `acl.sty` and `acl_natbib.bst` with official current ACL files.
- [ ] Compile in official ACL review mode.
- [ ] Keep content at or below eight pages before Limitations.
- [ ] Keep Limitations after Conclusion and before References.
- [ ] Add AI-assistance disclosure in Acknowledgements and Responsible NLP Checklist.
- [ ] Ensure every supplement and repository link is anonymized.
- [ ] Complete all author OpenReview profiles and designate a qualified service contributor.
