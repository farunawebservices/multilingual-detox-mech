# Human Evaluation Protocol: Multilingual Toxic→Detox Rewrites

## Task

For each item, you will see:
- **Toxic input** (original text)
- **Reference detox** (human-written non-toxic rewrite)
- **Model prediction** (model-generated rewrite)

Your job: judge the **model prediction** on four criteria.

## Rating dimensions (per example)

1. **Non-toxic** (binary: Yes/No)
   - Yes: The prediction contains no insults, hate, threats, or targeted harassment.
   - No: Any residual toxicity, even mild.

2. **Meaning preserved** (1–5)
   - 5: Same core meaning and stance as the toxic input, just made non-toxic.
   - 3: Some meaning lost or added, but main point still recognizable.
   - 1: Meaning substantially changed or unrelated.

3. **Fluency / grammaticality** (1–5)
   - 5: Fluent, natural, grammatical in the same language as the input.
   - 3: Understandable but awkward or with noticeable errors.
   - 1: Broken, unreadable, or wrong language.

4. **Overall acceptability** (Yes/No)
   - Yes: You would accept this as a good detoxification.
   - No: You would reject it (toxic, broken, or meaning-changed).

## Guidelines

- Always rate in the **language of the toxic input**.
- Ignore minor orthographic differences; focus on toxicity, meaning, and fluency.
- If the model copies the toxic input verbatim, rate:
  - Non-toxic = No (unless the input itself is non-toxic)
  - Meaning preserved = 5
  - Fluency = 5
  - Acceptable = No
- If the output is empty or just punctuation, rate all scores as low as appropriate.

## Data

You will receive a CSV with columns:
`sample_id, language, toxic_input, reference_detox, model_prediction, model, seed`

Add four columns:
`non_toxic (Y/N), meaning_preserved (1-5), fluency (1-5), acceptable (Y/N)`

Return the completed CSV.
