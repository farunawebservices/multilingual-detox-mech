from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset

ROOT = Path("/workspace/multilingual-detox-mech")
DATA = ROOT / "data"
OUT = DATA / "external" / "afrihate"

YO_EXISTING = DATA / "yoruba_pairs.csv"
XH_EXISTING = DATA / "isixhosa_pairs.csv"

OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "yor": {
        "language": "Yorùbá",
        "existing_path": YO_EXISTING,
        "output_path": OUT / "afrihate_yor_toxic_candidates_200.csv",
    },
    "xho": {
        "language": "isiXhosa",
        "existing_path": XH_EXISTING,
        "output_path": OUT / "afrihate_xho_toxic_candidates_200.csv",
    },
}

TOXIC_LABELS = {"Abuse", "Hate"}

def normalize(text):
    return " ".join(str(text).casefold().split())

def load_existing_inputs(path):
    df = pd.read_csv(path)
    input_col = "Toxic Sentence (Input)"
    if input_col not in df.columns:
        raise SystemExit(f"Expected column missing in {path}: {input_col}")
    return set(df[input_col].dropna().map(normalize))

def sanitize_for_processing(text):
    text = str(text)
    text = text.replace("@username", "@USER")
    text = text.replace("##url", "URL")
    return " ".join(text.split())

for config, settings in TARGETS.items():
    print(f"\nLoading AfriHate config: {config}")

    dataset_dict = load_dataset(
        "afrihate/afrihate",
        config,
        token=True,
    )

    all_splits = []
    for split_name, split in dataset_dict.items():
        frame = split.to_pandas()
        frame["source_split"] = split_name
        all_splits.append(frame)

    df = pd.concat(all_splits, ignore_index=True)

    required = {"id", "tweet", "label", "length"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Unexpected schema for {config}; missing {missing}")

    df["tweet"] = df["tweet"].fillna("").astype(str).str.strip()
    df = df[df["tweet"] != ""].copy()
    df = df[df["label"].isin(TOXIC_LABELS)].copy()

    df["normalized_tweet"] = df["tweet"].map(normalize)
    df = df.drop_duplicates(subset=["normalized_tweet"]).copy()

    existing = load_existing_inputs(settings["existing_path"])
    df["existing_parallel_overlap"] = df["normalized_tweet"].isin(existing)
    df = df[~df["existing_parallel_overlap"]].copy()

    df["toxic_input_raw"] = df["tweet"]
    df["toxic_input_processing"] = df["tweet"].map(sanitize_for_processing)
    df["source_dataset"] = "AfriHate"
    df["source_config"] = config
    df["language"] = settings["language"]
    df["phase1_status"] = "raw_native_annotated_toxic"

    # Sampling reproducibly across both labels, without manufacturing text.
    hate = df[df["label"] == "Hate"].sample(
        n=min(50, (df["label"] == "Hate").sum()),
        random_state=42,
    )
    remaining = df.drop(index=hate.index)
    abuse_needed = 200 - len(hate)
    abuse = remaining[remaining["label"] == "Abuse"].sample(
        n=min(abuse_needed, (remaining["label"] == "Abuse").sum()),
        random_state=42,
    )

    selected = pd.concat([hate, abuse], ignore_index=True)

    if len(selected) < 200:
        remainder = df.drop(index=selected.index, errors="ignore")
        selected = pd.concat(
            [selected, remainder.head(200 - len(selected))],
            ignore_index=True,
        )

    selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)

    output_cols = [
        "source_dataset",
        "source_config",
        "source_split",
        "id",
        "language",
        "label",
        "length",
        "toxic_input_raw",
        "toxic_input_processing",
        "existing_parallel_overlap",
        "phase1_status",
    ]

    selected[output_cols].to_csv(
        settings["output_path"],
        index=False,
        encoding="utf-8",
    )

    print("Available non-overlapping toxic rows:", len(df))
    print("Exported rows:", len(selected))
    print("Label distribution:")
    print(selected["label"].value_counts().to_string())
    print("Output:", settings["output_path"])