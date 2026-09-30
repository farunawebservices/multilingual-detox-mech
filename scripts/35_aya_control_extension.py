#!/usr/bin/env python3
"""
Generate two post-selection frozen-test controls for the locked Aya steering run.

Controls:
- negative_english_transfer: apply -alpha * English train-only direction.
- orthogonal_random_direction: apply +alpha * a seeded random vector orthogonal
  to the English train-only direction.

This script reads the committed locked test configuration and saved train-only
directions. It does not estimate directions, run development selection, or modify
existing baseline/transfer/within-language/random/shuffled outputs.
"""

from __future__ import annotations

import hashlib
import json
import random
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
RUN_DIR = Path("artifacts/confirmatory_aya_2026_09_30/primary_seed42")
TARGET_LANGS = ("xh", "yo")
RANDOM_SEED = 2042
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = (
    "Rewrite toxic text to be non-toxic while preserving its meaning "
    "and the original language. Return only the rewritten text."
)


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_example_id(row_index: int, row: pd.Series) -> str:
    payload = (
        f"test|{row_index}|{row['language']}|"
        f"{row['toxic_input']}|{row['detox_output']}"
    ).encode("utf-8")
    return f"test-{row['language']}-{row_index:05d}-{sha256_bytes(payload)[:12]}"


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


def load_test_data() -> pd.DataFrame:
    path = DATA_ROOT / "test.csv"
    df = pd.read_csv(path, keep_default_na=False).copy()
    df["row_index"] = np.arange(len(df))
    df["example_id"] = [
        stable_example_id(int(i), row)
        for i, row in df.iterrows()
    ]
    return (
        df[df["language"].isin(TARGET_LANGS)]
        .sort_values(["language", "row_index"])
        .reset_index(drop=True)
    )


def load_model():
    adapter_path = ADAPTER_ROOT / "seed_42" / "adapter"
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
    return model, tokenizer


def orthogonal_random_direction(
    steering_direction: torch.Tensor,
    random_seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(random_seed)
    random_vector = torch.randn(
        steering_direction.shape,
        generator=generator,
        dtype=torch.float32,
    )
    steering_direction = steering_direction.float().cpu()
    random_vector = random_vector - (
        torch.dot(random_vector, steering_direction) * steering_direction
    )
    if random_vector.norm().item() == 0.0:
        raise RuntimeError("Orthogonal random direction has zero norm.")
    return random_vector / random_vector.norm()


def generate_one(
    model,
    tokenizer,
    toxic_text: str,
    layer: int,
    direction: torch.Tensor,
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
    direction: torch.Tensor,
) -> pd.DataFrame:
    rows = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
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
                "language": row["language"],
                "language_name": row.get("language_name", ""),
                "toxic_input": row["toxic_input"],
                "detox_output": row["detox_output"],
                "condition": condition,
                "layer": layer,
                "alpha": alpha,
                "prediction": prediction,
            }
        )
        if position % 25 == 0 or position == len(df):
            print(f"  {condition}: {position}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


def add_metrics(df: pd.DataFrame, embedder: SentenceTransformer) -> pd.DataFrame:
    out = df.copy()
    predictions = out["prediction"].astype(str).tolist()
    references = out["detox_output"].astype(str).tolist()
    toxic_inputs = out["toxic_input"].astype(str).tolist()

    pred_emb = embedder.encode(
        predictions, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    ref_emb = embedder.encode(
        references, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    input_emb = embedder.encode(
        toxic_inputs, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )

    out["labse_to_reference"] = np.sum(pred_emb * ref_emb, axis=1)
    out["labse_to_input"] = np.sum(pred_emb * input_emb, axis=1)
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
    out["output_words"] = [len(normalize_text(pred).split()) for pred in predictions]
    return out


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    lock_path = RUN_DIR / "config" / "locked_test_configuration.json"
    direction_path = RUN_DIR / "directions" / "train_only_directions_layer_16.pt"
    test_dir = RUN_DIR / "test"
    metrics_dir = RUN_DIR / "metrics"

    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    if not direction_path.exists():
        raise FileNotFoundError(direction_path)

    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    layer = int(locked["selected_layer"])
    alpha = float(locked["selected_alpha"])

    if layer != 16 or alpha != 2.0:
        raise RuntimeError(
            f"Expected locked layer=16, alpha=2.0; got layer={layer}, alpha={alpha}"
        )

    payload = torch.load(direction_path, map_location="cpu", weights_only=False)
    en_direction = payload["directions"]["en"].float()
    en_direction = en_direction / en_direction.norm()
    orthogonal_direction = orthogonal_random_direction(en_direction, RANDOM_SEED)

    cosine = float(torch.dot(en_direction, orthogonal_direction))
    print("English/orthogonal cosine:", cosine)
    if abs(cosine) > 1e-5:
        raise RuntimeError(f"Orthogonal control is not orthogonal: cosine={cosine}")

    test_df = load_test_data()
    if len(test_df) != 133:
        raise RuntimeError(f"Expected 133 xh/yo test rows; found {len(test_df)}")

    for condition in ("negative_english_transfer", "orthogonal_random_direction"):
        for suffix in ("_generations.csv", "_metrics.csv"):
            path = test_dir / f"{condition}{suffix}"
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {path}"
                )

    print("Loading model...")
    model, tokenizer = load_model()
    print("Loading LaBSE...")
    embedder = SentenceTransformer("sentence-transformers/LaBSE")

    specifications = {
        "negative_english_transfer": {
            "direction": en_direction,
            "alpha": -alpha,
        },
        "orthogonal_random_direction": {
            "direction": orthogonal_direction,
            "alpha": alpha,
        },
    }

    metrics_frames = []

    for condition, spec in specifications.items():
        print(f"Generating {condition}...")
        generated = generate_condition(
            model=model,
            tokenizer=tokenizer,
            df=test_df,
            layer=layer,
            alpha=spec["alpha"],
            condition=condition,
            direction=spec["direction"],
        )
        generated.to_csv(
            test_dir / f"{condition}_generations.csv",
            index=False,
            encoding="utf-8",
        )

        metrics = add_metrics(generated, embedder)
        metrics.to_csv(
            test_dir / f"{condition}_metrics.csv",
            index=False,
            encoding="utf-8",
        )
        metrics_frames.append(metrics)

    extension_metrics = pd.concat(metrics_frames, ignore_index=True)
    extension_metrics.to_csv(
        metrics_dir / "control_extension_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    summary = (
        extension_metrics.groupby(["condition", "language"], as_index=False)
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
    summary.to_csv(
        metrics_dir / "control_extension_summary.csv",
        index=False,
        encoding="utf-8",
    )

    run_manifest = {
        "model_id": MODEL_ID,
        "adapter_seed": 42,
        "locked_configuration": locked,
        "direction_file": str(direction_path),
        "random_seed": RANDOM_SEED,
        "orthogonal_cosine_with_english": cosine,
        "test_examples": len(test_df),
        "conditions": list(specifications),
    }
    (RUN_DIR / "config" / "control_extension_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(run_manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
