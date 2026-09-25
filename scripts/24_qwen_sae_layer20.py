#!/usr/bin/env python3
"""
Stage 24: Train Sparse Autoencoder (SAE) on Qwen2.5-7B layer 20 activations.

Goal (RQ2: feature-level interpretability):
- Learn a sparse overcomplete basis for toxic→detox transformation.
- Identify detox-selective vs toxic-selective features.
- Save SAE weights and top features for later steering.

Uses TRAIN split only (no leakage into dev/test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT = Path("results/training_v2_400plus/seed_2024")
TRAIN_ROOT = Path("results/training_qwen2_5_7b")
OUTPUT_ROOT = Path("results/mechanistic/qwen_sae")

TARGET_LAYER = 20
SAE_HIDDEN = 8192  # 2× model dimension (3584 → 8192)
SPARSITY_COEF = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_TRAIN_EXAMPLES = 2000  # cap for training speed


SYSTEM_PROMPT = (
    "Rewrite toxic text to be non-toxic while preserving its meaning "
    "and the original language. Return only the rewritten text."
)


def make_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def collect_activations(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    layer_idx: int,
    max_examples: int | None = None,
):
    """
    Collect last-token hidden states for toxic and detox inputs.
    Returns concatenated activations [n_toxic + n_detox, d_model].
    """
    if max_examples is not None:
        dataframe = dataframe.iloc[:max_examples].reset_index(drop=True)

    acts = []

    model.eval()
    with torch.inference_mode():
        for _, row in dataframe.iterrows():
            for text in [str(row["toxic_input"]), str(row["detox_output"])]:
                prompt = make_prompt(tokenizer, text)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )

                hidden = (
                    outputs.hidden_states[layer_idx][0, -1, :]
                    .float()
                    .cpu()
                )
                acts.append(hidden)

    return torch.stack(acts, dim=0)  # [N, d_model]


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h


def train_sae(
    activations: torch.Tensor,
    d_hidden: int,
    sparsity_coef: float,
    num_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    d_model = activations.shape[1]
    sae = SparseAutoencoder(d_model, d_hidden).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_l1 = 0.0
        n_batches = 0

        for (batch,) in loader:
            x = batch
            x_hat, h = sae(x)

            recon_loss = F.mse_loss(x_hat, x)
            l1_loss = h.mean()
            loss = recon_loss + sparsity_coef * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_recon += float(recon_loss)
            total_l1 += float(l1_loss)
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_l1 = total_l1 / n_batches
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"loss={avg_loss:.6f}, recon={avg_recon:.6f}, L1={avg_l1:.6f}"
        )

    return sae


def analyze_features(
    sae: SparseAutoencoder,
    toxic_act: torch.Tensor,
    detox_act: torch.Tensor,
    device: torch.device,
    top_k: int = 50,
):
    sae.eval()
    with torch.inference_mode():
        toxic_h = F.relu(sae.encoder(toxic_act.to(device)))
        detox_h = F.relu(sae.encoder(detox_act.to(device)))

        toxic_mean = toxic_h.mean(dim=0)
        detox_mean = detox_h.mean(dim=0)
        diff = detox_mean - toxic_mean

        # Detox-selective features (positive diff)
        detox_selective = diff.argsort(descending=True)[:top_k]
        # Toxic-selective features (negative diff)
        toxic_selective = diff.argsort()[:top_k]

    return {
        "detox_selective": detox_selective.cpu().tolist(),
        "toxic_selective": toxic_selective.cpu().tolist(),
        "detox_mean": detox_mean.cpu().tolist(),
        "toxic_mean": toxic_mean.cpu().tolist(),
        "diff": diff.cpu().tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed

    adapter_path = TRAIN_ROOT / f"seed_{seed}" / "adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    train_path = DATA_ROOT / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    out_dir = OUTPUT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path, keep_default_na=False)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", seed)
    print("Train examples (raw):", len(train_df))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen base model in 4-bit...")
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

    print(f"Collecting layer {TARGET_LAYER} activations (train split)...")
    all_acts = collect_activations(
        model,
        tokenizer,
        train_df,
        TARGET_LAYER,
        max_examples=MAX_TRAIN_EXAMPLES,
    )
    print(f"Total activations: {all_acts.shape}")

    # Split into toxic/detox halves (even rows = toxic, odd = detox by construction)
    n = all_acts.shape[0]
    toxic_act = all_acts[0::2]
    detox_act = all_acts[1::2]
    print(f"Toxic: {toxic_act.shape}, Detox: {detox_act.shape}")

    print("\nTraining SAE...")
    device = torch.device("cuda")
    sae = train_sae(
        all_acts,
        d_hidden=SAE_HIDDEN,
        sparsity_coef=SPARSITY_COEF,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=device,
    )

    print("\nAnalyzing SAE features...")
    feature_analysis = analyze_features(
        sae, toxic_act, detox_act, device, top_k=50
    )

    # Save SAE weights and analysis
    torch.save(
        {
            "encoder_weight": sae.encoder.weight.detach().cpu(),
            "decoder_weight": sae.decoder.weight.detach().cpu(),
            "d_model": all_acts.shape[1],
            "d_hidden": SAE_HIDDEN,
            "target_layer": TARGET_LAYER,
            "seed": seed,
            "feature_analysis": feature_analysis,
        },
        out_dir / "sae_layer20.pt",
    )

    # Save a compact summary JSON
    summary = {
        "model_id": MODEL_ID,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "target_layer": TARGET_LAYER,
        "sae_hidden": SAE_HIDDEN,
        "sparsity_coef": SPARSITY_COEF,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_train_examples": MAX_TRAIN_EXAMPLES,
        "total_activations": int(all_acts.shape[0]),
        "toxic_activations": int(toxic_act.shape[0]),
        "detox_activations": int(detox_act.shape[0]),
        "top_detox_features": feature_analysis["detox_selective"][:20],
        "top_toxic_features": feature_analysis["toxic_selective"][:20],
    }

    (out_dir / "sae_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {out_dir / 'sae_layer20.pt'}")
    print(f"  {out_dir / 'sae_summary.json'}")

    del sae
    del model
    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
