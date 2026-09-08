# AfriHate-derived Toxic/Detox Candidates for Human Review

This directory contains candidate toxic → detoxified sentence pairs for **Yorùbá** and **isiXhosa**, derived from the [AfriHate](https://huggingface.co/datasets/afrihate/afrihate) hate-speech/abusive-language dataset.

These candidates are intended for **human review** before being added to the final fine-tuning corpus for multilingual text detoxification.

## Files

- `candidates_for_review.csv` – Full set of candidate pairs for human annotation.

## Dataset statistics

- **Total candidate pairs:** 665  
  - **Yorùbá:** 330  
  - **isiXhosa:** 335  

All candidates are from AfriHate tweets annotated as **Abuse** or **Hate** by native speakers.

## Data source

- **AfriHate**: A multilingual collection of hate speech and abusive language datasets for African languages, annotated by native speakers.  
  - HF repo: https://huggingface.co/datasets/afrihate/afrihate  
  - Languages used here: `yor` (Yorùbá), `xho` (isiXhosa)

When using this data, please cite the AfriHate dataset according to its repository instructions.

## How the candidates were created

1. **Source selection**  
   - Loaded all Yorùbá and isiXhosa tweets from AfriHate (train/dev/test splits).  
   - Filtered to labels: `Abuse` or `Hate`.  
   - Excluded any rows overlapping with the existing human parallel corpus.

2. **Detoxification**  
   - Toxic posts were rewritten into non-toxic versions using:
     - `openai/gpt-5` for Yorùbá  
     - `openai/gpt-5-mini` for isiXhosa  
   - The prompt instructed the model to:
     - preserve topic, target, and core meaning,
     - remove/soften insults, profanity, hostility, and threats,
     - keep natural code-switching only when necessary,
     - avoid adding new facts or commentary.

3. **Automatic quality filtering**  
   - Each (toxic, detox) pair was judged by `openai/gpt-4o` with a structured JSON schema.  
   - Only pairs with:
     - `source_decision = suitable` and  
     - `final_decision ∈ {accepted, needs_edit}`  
     were kept as candidates.

This process yielded **665 candidate pairs** (330 Yorùbá, 335 isiXhosa).

## CSV schema

`candidates_for_review.csv` columns:

- `language` – `Yorùbá` or `isiXhosa`
- `source_id` – AfriHate tweet ID (e.g. `train_yoruba_01273`)
- `toxic` – Original toxic tweet text (AfriHate)
- `detox` – GPT‑5 detoxified candidate
- `afrihate_label` – Original AfriHate label (`Abuse` or `Hate`)
- `judge_decision` – GPT‑4 judgment: `accepted` or `needs_edit`
- `human_decision` – **To be filled by reviewers**: `accept`, `edit`, or `reject`
- `human_comments` – Optional short notes from reviewers
- `edited_detox` – If `human_decision = edit`, the reviewer’s corrected detox version

## Human review instructions

For each row, a native or proficient speaker should:

1. Read `toxic` and `detox`.
2. Set `human_decision` to one of:
   - `accept`  
     - `toxic` is genuinely toxic/harmful in the target language.  
     - `detox` is natural, non-toxic, and preserves the core meaning.
   - `edit`  
     - The pair is basically usable but `detox` needs a small fix (grammar, meaning, tone).  
     - Fill `edited_detox` with the corrected detox sentence.
   - `reject`  
     - `toxic` is not really toxic, is mostly another language, or is unsafe (identity-hate, severe violence, explicit sexual content).  
     - Or `detox` is unnatural, changes the meaning too much, or remains toxic.
3. Optionally add a brief note in `human_comments` (e.g. “too much English”, “meaning changed”, “good natural detox”).

After review, rows with `human_decision` in `{accept, edit}` (using `edited_detox` where applicable) can be merged into the final parallel training corpus.

## Intended use

- These candidates are **not yet final ground truth**.  
- They are meant to be:
  - human-reviewed,
  - possibly edited,
  - then merged with existing human-written parallel data to train and evaluate detoxification models for Yorùbá and isiXhosa.

## License and citation

- The candidate pairs themselves are released under the same license terms as the underlying AfriHate data (check the AfriHate repository for details).  
- When publishing work that uses this data, cite:
  - the AfriHate dataset (per their README), and
  - your own paper/project describing the detoxification pipeline and human review process.