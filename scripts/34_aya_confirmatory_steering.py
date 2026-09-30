#!/usr/bin/env python3
"""
Confirmatory activation steering for Aya-23-8B.

Protocol:
- Direction estimation: train split only.
- Layer/strength selection: target-language development data only.
- Final evaluation: frozen target-language test data only.
- Conditions: baseline, English-transfer, within-language, random-direction,
  shuffled-pair direction.
- Decoding: deterministic greedy decoding.
- All per-example generations and metrics are saved.

This script intentionally does not make human-evaluation or validated-detoxification
claims. It produces the paired output package required for subsequent blinded rating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "CohereLabs/aya-23-8B"
DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
ADAPTER_ROOT = Path("results/training_aya_23_8b")
RUN_ROOT = Path("artifacts/confirmatory_aya_2026_09_30")
TARGET_LANGS = ("xh", "yo")
SOURCE_LANG = "en"
DEFAULT_LAYERS = (16, 20)
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "Rewrite toxic text to be non-toxic while preserving its meaning "
    "and the original language. Return only the rewritten text."
)


def parse_csv_list(value: str, cast):
    return tuple(cast(x.strip()) for x in value.split(",") if x.strip())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_example_id(split: str, row_index: int, row: pd.Series) -> str:
    payload = (
        f"{split}|{row_index}|{row['language']}|"
        f"{row['toxic_input']}|{row['detox_output']}"
    ).encode("utf-8")
    return f"{split}-{row['language']}-{row_index:05d}-{sha256_bytes(payload)[:12]}"


def load_split(split: str) -> pd.DataFrame:
    path = DATA_ROOT / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    required = {"language", "toxic_input", "detox_output"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["row_index"] = np.arange(len(df))
    df["example_id"] = [
        stable_example_id(split, int(i), row)
        for i, row in df.iterrows()
    ]
    return df


def target_subset(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["language"].isin(TARGET_LANGS)]
        .copy()
        .sort_values(["language", "row_index"])
        .reset_index(drop=True)
    )


def language_subset(df: pd.DataFrame, language: str) -> pd.DataFrame:
    return (
        df[df["language"] == language]
        .copy()
        .sort_values("row_index")
        .reset_index(drop=True)
    )


def make_prompt(tokenizer, toxic_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(toxic_text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_decoder_layer(model, layer_idx: int):
    return model.base_model.model.model.layers[layer_idx]


def load_model(seed: int):
    adapter_path = ADAPTER_ROOT / f"seed_{seed}" / "adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, adapter_path


def collect_last_token_states(
    model,
    tokenizer,
    df: pd.DataFrame,
    layer: int,
    text_column: str,
) -> torch.Tensor:
    states = []
    with torch.inference_mode():
        for position, (_, row) in enumerate(df.iterrows(), start=1):
            prompt = make_prompt(tokenizer, str(row[text_column]))
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            ).to(model.device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            states.append(
                outputs.hidden_states[layer][0, -1, :].float().cpu()
            )

            if position % 50 == 0 or position == len(df):
                print(
                    f"  layer={layer} column={text_column}: "
                    f"{position}/{len(df)}",
                    flush=True,
                )

    return torch.stack(states)


def direction_from_paired_rows(
    model,
    tokenizer,
    df: pd.DataFrame,
    layer: int,
) -> torch.Tensor:
    toxic = collect_last_token_states(model, tokenizer, df, layer, "toxic_input")
    detox = collect_last_token_states(model, tokenizer, df, layer, "detox_output")
    vector = detox.mean(dim=0) - toxic.mean(dim=0)
    return vector / vector.norm()


def shuffled_pair_direction(
    model,
    tokenizer,
    df: pd.DataFrame,
    layer: int,
    random_seed: int,
) -> torch.Tensor:
    toxic = collect_last_token_states(model, tokenizer, df, layer, "toxic_input")
    detox = collect_last_token_states(model, tokenizer, df, layer, "detox_output")
    generator = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(len(detox), generator=generator)
    vector = detox[perm].mean(dim=0) - toxic.mean(dim=0)
    if torch.allclose(vector.norm(), torch.tensor(0.0)):
        raise RuntimeError("Shuffled direction has zero norm.")
    return vector / vector.norm()


def random_direction_like(vector: torch.Tensor, random_seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(random_seed)
    direction = torch.randn(
        vector.shape,
        generator=generator,
        dtype=torch.float32,
    )
    return direction / direction.norm()


def generate_one(
    model,
    tokenizer,
    toxic_text: str,
    layer: int,
    direction: torch.Tensor | None,
    alpha: float,
) -> str:
    prompt = make_prompt(tokenizer, toxic_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(model.device)

    handle = None
    if direction is not None and alpha != 0.0:
        vector = direction.to(model.device, dtype=torch.bfloat16)

        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * vector
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = get_decoder_layer(model, layer).register_forward_hook(hook_fn)

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()

    prompt_length = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def generate_condition(
    model,
    tokenizer,
    df: pd.DataFrame,
    layer: int,
    alpha: float,
    condition: str,
    direction_by_lang: dict[str, torch.Tensor | None],
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        language = str(row["language"])
        direction = direction_by_lang.get(language)
        prediction = generate_one(
            model=model,
            tokenizer=tokenizer,
            toxic_text=str(row["toxic_input"]),
            layer=layer,
            direction=direction,
            alpha=alpha,
        )
        rows.append(
            {
                "example_id": row["example_id"],
                "row_index": int(row["row_index"]),
                "language": language,
                "language_name": row.get("language_name", ""),
                "toxic_input": row["toxic_input"],
                "detox_output": row["detox_output"],
                "condition": condition,
                "layer": int(layer),
                "alpha": float(alpha),
                "prediction": prediction,
            }
        )
        if position % 25 == 0 or position == len(df):
            print(
                f"  {condition}: {position}/{len(df)}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def add_automatic_metrics(df: pd.DataFrame, embedder: SentenceTransformer) -> pd.DataFrame:
    out = df.copy()
    predictions = out["prediction"].astype(str).tolist()
    references = out["detox_output"].astype(str).tolist()
    toxic_inputs = out["toxic_input"].astype(str).tolist()

    prediction_embeddings = embedder.encode(
        predictions,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    reference_embeddings = embedder.encode(
        references,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    input_embeddings = embedder.encode(
        toxic_inputs,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    out["labse_to_reference"] = np.sum(
        prediction_embeddings * reference_embeddings,
        axis=1,
    )
    out["labse_to_input"] = np.sum(
        prediction_embeddings * input_embeddings,
        axis=1,
    )
    out["chrf_to_reference"] = [
        sacrebleu.sentence_chrf(pred, [ref]).score / 100.0
        for pred, ref in zip(predictions, references)
    ]
    out["copy_input"] = [
        normalize_text(pred) == normalize_text(src)
        for pred, src in zip(predictions, toxic_inputs)
    ]
    out["copy_reference"] = [
        normalize_text(pred) == normalize_text(ref)
        for pred, ref in zip(predictions, references)
    ]
    out["empty_output"] = [not bool(normalize_text(pred)) for pred in predictions]
    out["output_chars"] = [len(pred) for pred in predictions]
    out["output_words"] = [
        len(normalize_text(pred).split()) for pred in predictions
    ]
    return out


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["condition", "language"], as_index=False)
        .agg(
            n_examples=("example_id", "count"),
            chrf_to_reference=("chrf_to_reference", "mean"),
            labse_to_reference=("labse_to_reference", "mean"),
            labse_to_input=("labse_to_input", "mean"),
            copy_input=("copy_input", "mean"),
            copy_reference=("copy_reference", "mean"),
            empty_output=("empty_output", "mean"),
            output_chars=("output_chars", "mean"),
            output_words=("output_words", "mean"),
            unique_output_rate=("prediction", lambda x: x.nunique() / len(x)),
        )
    )


def choose_dev_configuration(
    summaries: pd.DataFrame,
    candidate_layers: tuple[int, ...],
    candidate_strengths: tuple[float, ...],
) -> dict:
    pooled = (
        summaries.groupby(["layer", "alpha"], as_index=False)
        .agg(
            chrf_to_reference=("chrf_to_reference", "mean"),
            labse_to_reference=("labse_to_reference", "mean"),
            labse_to_input=("labse_to_input", "mean"),
            copy_input=("copy_input", "mean"),
            empty_output=("empty_output", "mean"),
        )
    )

    baseline = pooled[pooled["alpha"] == 0.0]
    if baseline.empty:
        raise RuntimeError("Development baseline rows are missing.")

    baseline_by_layer = baseline.set_index("layer").to_dict(orient="index")
    eligible = []

    for _, row in pooled.iterrows():
        layer = int(row["layer"])
        alpha = float(row["alpha"])
        base = baseline_by_layer[layer]

        if row["empty_output"] > base["empty_output"]:
            continue
        if row["copy_input"] > base["copy_input"] + 0.05:
            continue
        if row["labse_to_reference"] < base["labse_to_reference"] - 0.05:
            continue

        score = (
            0.70 * row["chrf_to_reference"]
            + 0.20 * row["labse_to_reference"]
            - 0.10 * row["labse_to_input"]
        )
        eligible.append(
            {
                "layer": layer,
                "alpha": alpha,
                "selection_score": float(score),
                "chrf_to_reference": float(row["chrf_to_reference"]),
                "labse_to_reference": float(row["labse_to_reference"]),
                "labse_to_input": float(row["labse_to_input"]),
                "copy_input": float(row["copy_input"]),
                "empty_output": float(row["empty_output"]),
            }
        )

    if not eligible:
        raise RuntimeError("No development configuration passed the safety filters.")

    best = max(eligible, key=lambda x: x["selection_score"])
    return {
        "selected": best,
        "all_pooled": pooled.to_dict(orient="records"),
        "eligible": eligible,
        "candidate_layers": list(candidate_layers),
        "candidate_strengths": list(candidate_strengths),
        "selection_rule": (
            "Reject configurations with elevated empty-output rate, input-copy "
            "rate > baseline + 0.05, or LaBSE-to-reference < baseline - 0.05; "
            "among remaining configurations maximize "
            "0.70*chrF_ref + 0.20*LaBSE_ref - 0.10*LaBSE_input."
        ),
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--layers", default="16,20")
    parser.add_argument("--strengths", default="0.0,0.25,0.5,1.0,2.0")
    parser.add_argument("--run-name", default="primary_seed42")
    args = parser.parse_args()

    candidate_layers = parse_csv_list(args.layers, int)
    candidate_strengths = parse_csv_list(args.strengths, float)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    run_dir = RUN_ROOT / args.run_name
    config_dir = run_dir / "config"
    direction_dir = run_dir / "directions"
    dev_dir = run_dir / "dev"
    test_dir = run_dir / "test"
    metrics_dir = run_dir / "metrics"
    for directory in (config_dir, direction_dir, dev_dir, test_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    dev_df = target_subset(load_split("dev"))
    test_df = target_subset(load_split("test"))

    split_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "adapter_seed": args.seed,
        "data_root": str(DATA_ROOT),
        "train_sha256": sha256_file(DATA_ROOT / "train.csv"),
        "dev_sha256": sha256_file(DATA_ROOT / "dev.csv"),
        "test_sha256": sha256_file(DATA_ROOT / "test.csv"),
        "train_total": len(train_df),
        "dev_target_total": len(dev_df),
        "test_target_total": len(test_df),
        "dev_target_by_language": dev_df["language"].value_counts().sort_index().to_dict(),
        "test_target_by_language": test_df["language"].value_counts().sort_index().to_dict(),
        "candidate_layers": list(candidate_layers),
        "candidate_strengths": list(candidate_strengths),
        "source_direction_language": SOURCE_LANG,
        "target_languages": list(TARGET_LANGS),
        "random_seed": RANDOM_SEED,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": 1.05,
        },
    }
    write_json(config_dir / "split_manifest.json", split_manifest)

    print("Loading Aya model and adapter...", flush=True)
    model, tokenizer, adapter_path = load_model(args.seed)
    split_manifest["adapter_path"] = str(adapter_path)
    write_json(config_dir / "split_manifest.json", split_manifest)

    train_en = language_subset(train_df, SOURCE_LANG)
    train_xh = language_subset(train_df, "xh")
    train_yo = language_subset(train_df, "yo")

    directions: dict[int, dict[str, torch.Tensor]] = {}

    print("Estimating train-only directions...", flush=True)
    for layer in candidate_layers:
        print(f"=== Direction layer {layer} ===", flush=True)
        en_direction = direction_from_paired_rows(model, tokenizer, train_en, layer)
        xh_direction = direction_from_paired_rows(model, tokenizer, train_xh, layer)
        yo_direction = direction_from_paired_rows(model, tokenizer, train_yo, layer)
        shuffled_direction = shuffled_pair_direction(
            model, tokenizer, train_en, layer, RANDOM_SEED + layer
        )
        random_direction = random_direction_like(en_direction, RANDOM_SEED + 1000 + layer)

        directions[layer] = {
            "en": en_direction,
            "xh": xh_direction,
            "yo": yo_direction,
            "shuffled_en": shuffled_direction,
            "random": random_direction,
        }

        torch.save(
            {
                "layer": layer,
                "split": "train_only",
                "source_counts": {
                    "en": len(train_en),
                    "xh": len(train_xh),
                    "yo": len(train_yo),
                },
                "directions": {name: value.cpu() for name, value in directions[layer].items()},
            },
            direction_dir / f"train_only_directions_layer_{layer}.pt",
        )

    if args.phase in ("dev", "all"):
        print("Running development selection sweep...", flush=True)
        dev_records = []

        for layer in candidate_layers:
            direction_by_lang = {"xh": directions[layer]["en"], "yo": directions[layer]["en"]}
            for alpha in candidate_strengths:
                condition = f"dev_en_transfer_l{layer}_a{alpha:g}"
                raw_path = dev_dir / f"{condition}.csv"
                generated = generate_condition(
                    model=model,
                    tokenizer=tokenizer,
                    df=dev_df,
                    layer=layer,
                    alpha=alpha,
                    condition=condition,
                    direction_by_lang=direction_by_lang,
                    output_path=raw_path,
                )
                metrics = add_automatic_metrics(
                    generated,
                    SentenceTransformer("sentence-transformers/LaBSE"),
                )
                metrics["layer"] = layer
                metrics["alpha"] = alpha
                metrics_path = dev_dir / f"{condition}_metrics.csv"
                metrics.to_csv(metrics_path, index=False, encoding="utf-8")
                dev_records.append(metrics)

        dev_metrics = pd.concat(dev_records, ignore_index=True)
        dev_metrics.to_csv(dev_dir / "all_dev_metrics.csv", index=False, encoding="utf-8")
        dev_summary = (
            dev_metrics.groupby(["layer", "alpha", "language"], as_index=False)
            .agg(
                n_examples=("example_id", "count"),
                chrf_to_reference=("chrf_to_reference", "mean"),
                labse_to_reference=("labse_to_reference", "mean"),
                labse_to_input=("labse_to_input", "mean"),
                copy_input=("copy_input", "mean"),
                empty_output=("empty_output", "mean"),
            )
        )
        dev_summary.to_csv(dev_dir / "dev_summary.csv", index=False, encoding="utf-8")

        selection = choose_dev_configuration(
            dev_summary,
            candidate_layers=candidate_layers,
            candidate_strengths=candidate_strengths,
        )
        write_json(config_dir / "dev_selection.json", selection)
        print("Development selection:", json.dumps(selection["selected"], indent=2), flush=True)

    if args.phase == "dev":
        return

    selection_path = config_dir / "dev_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing {selection_path}. Run --phase dev first, inspect and lock selection."
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_layer = int(selection["selected"]["layer"])
    selected_alpha = float(selection["selected"]["alpha"])

    locked_path = config_dir / "locked_test_configuration.json"

    if not locked_path.exists():
        raise FileNotFoundError(
            f"Missing locked configuration: {locked_path}. "
            "Create and commit it before frozen-test generation."
        )

    locked_selection = json.loads(
        locked_path.read_text(encoding="utf-8")
    )

    if (
        int(locked_selection["selected_layer"]) != selected_layer
        or float(locked_selection["selected_alpha"]) != selected_alpha
    ):
        raise RuntimeError(
            "Development selection does not match the committed "
            "locked test configuration."
        )

    print(
        f"Generating frozen test conditions at layer={selected_layer}, alpha={selected_alpha}",
        flush=True,
    )

    test_conditions = {
        "baseline": {
            "alpha": 0.0,
            "directions": {"xh": None, "yo": None},
        },
        "english_transfer": {
            "alpha": selected_alpha,
            "directions": {
                "xh": directions[selected_layer]["en"],
                "yo": directions[selected_layer]["en"],
            },
        },
        "within_language": {
            "alpha": selected_alpha,
            "directions": {
                "xh": directions[selected_layer]["xh"],
                "yo": directions[selected_layer]["yo"],
            },
        },
        "random_direction": {
            "alpha": selected_alpha,
            "directions": {
                "xh": directions[selected_layer]["random"],
                "yo": directions[selected_layer]["random"],
            },
        },
        "shuffled_pair": {
            "alpha": selected_alpha,
            "directions": {
                "xh": directions[selected_layer]["shuffled_en"],
                "yo": directions[selected_layer]["shuffled_en"],
            },
        },
    }

    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    all_test_metrics = []

    for condition, specification in test_conditions.items():
        raw_path = test_dir / f"{condition}_generations.csv"
        generated = generate_condition(
            model=model,
            tokenizer=tokenizer,
            df=test_df,
            layer=selected_layer,
            alpha=float(specification["alpha"]),
            condition=condition,
            direction_by_lang=specification["directions"],
            output_path=raw_path,
        )
        metrics = add_automatic_metrics(generated, embedder)
        metrics.to_csv(
            test_dir / f"{condition}_metrics.csv",
            index=False,
            encoding="utf-8",
        )
        all_test_metrics.append(metrics)

    combined = pd.concat(all_test_metrics, ignore_index=True)
    combined.to_csv(
        metrics_dir / "test_metrics_all_conditions.csv",
        index=False,
        encoding="utf-8",
    )
    summary = summarize_metrics(combined)
    summary.to_csv(
        metrics_dir / "test_metrics_summary.csv",
        index=False,
        encoding="utf-8",
    )

    run_summary = {
        "run_name": args.run_name,
        "model_id": MODEL_ID,
        "adapter_seed": args.seed,
        "selected_layer": selected_layer,
        "selected_alpha": selected_alpha,
        "test_examples": len(test_df),
        "test_languages": list(TARGET_LANGS),
        "conditions": list(test_conditions),
        "outputs": {
            "generations_dir": str(test_dir),
            "metrics_summary": str(metrics_dir / "test_metrics_summary.csv"),
            "combined_metrics": str(metrics_dir / "test_metrics_all_conditions.csv"),
        },
    }
    write_json(config_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
