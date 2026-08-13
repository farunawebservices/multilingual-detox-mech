#!/usr/bin/env python
"""Stage 3 -- Fine-tune google/mt5-base on the toxic -> detox task.

Trains on data/splits/train.csv and validates on data/splits/dev.csv, for seeds
42, 1337 and 2024. data/splits/test.csv is never opened; neither are the reserve
files. A path guard enforces this at runtime rather than relying on convention.

Usage
-----
    python scripts/03_train_mt5_detox.py --dry-run     # print the plan, train nothing
    python scripts/03_train_mt5_detox.py               # train all three seeds
    python scripts/03_train_mt5_detox.py --seeds 42    # train one seed

Outputs
-------
models/mt5_detox_baseline/seed<N>/        best checkpoint (by aggregate dev loss)
results/training/seed<N>/                 trainer logs
results/training/dev_loss_history.json    per-epoch, per-language dev loss, all seeds
results/training/training_summary.csv     best epoch and final dev loss per seed/language

Hyperparameter reasoning (judgment calls, since the brief leaves these open)
---------------------------------------------------------------------------
Learning rate 1e-4, AdamW, linear decay, 10% warmup.
    mT5 was pretrained on span corruption alone with no supervised task mixing, so it
    needs a higher learning rate than T5.1.1 to leave that regime; the widely used
    settings are 1e-3 with Adafactor or 5e-4 with AdamW. 1e-4 is deliberately at the
    conservative end of the workable band. Going lower (e.g. 3e-5) risks the classic
    mT5 failure where the model never adapts and emits <extra_id_0> sentinel tokens
    instead of text, which would look like a modelling result but is an optimisation
    artefact.

Precision bf16, never fp16.
    mT5 overflows to NaN under fp16 because of its gated-GELU activations. The A100 is
    compute capability 8.0, so bf16 is available and avoids the problem entirely.

Batch size 8 x 2 gradient accumulation = effective 16.
    1098 training rows give ~69 optimiser steps per epoch. The A100 could hold far more
    per step, but a larger batch would leave too few updates for the model to converge
    on a dataset this small. Small batches buy update steps, which is the binding
    constraint here, not memory.

Max epochs 20 with early stopping, patience 4.
    ~1373 steps at the ceiling. With 122 pairs per language mT5-base will overfit, so
    the epoch count is a ceiling rather than a target and early stopping is expected to
    end training well before it. If a run does reach the ceiling while still improving,
    that is reported rather than silently accepted.

Source length 96, target length 112.
    Measured, not guessed. On train+dev the source p99 is 73 tokens and the max is 90
    (Amharic), so 96 truncates nothing. For the target side the Stage 1 whole-corpus
    inventory shows a maximum of 102 tokens, again Amharic, so 112 leaves the test split
    untruncated too. Those corpus-level maxima come from the Stage 1 aggregate statistics,
    so no test row had to be read to establish them.

Uniform task prefix "detoxify: ", with no language tag.
    A language-tagged prefix would plant an explicit language-identity token in the
    encoder, and the Stage 7 language-identity probes would then largely be reading that
    token back rather than any representation the model builds from the text. Keeping the
    prefix language-neutral means the model has to infer language from content, which is
    what the probes are supposed to measure. The prefix occupies encoder positions 0-2;
    Stage 6 should therefore be aware of it when pooling activations.

Loss-only validation.
    Dev loss drives model selection. Generation metrics (STA/SIM/FL) are Stage 5's job,
    and computing them here would both slow every epoch and blur the boundary between
    the training stage and the evaluation stage.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODEL_DIR = REPO_ROOT / "models" / "mt5_detox_baseline"
LOG_DIR = REPO_ROOT / "results" / "training"

TASK_PREFIX = "detoxify: "

HP = {
    "learning_rate": 1e-4,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 32,
    "max_source_length": 96,
    "max_target_length": 112,
    "early_stopping_patience": 4,
    "metric_for_best_model": "eval_all_loss",
    "precision": "bf16",
    "optimizer": "adamw_torch",
}


def guarded_read(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Read a split CSV, refusing anything from the test split or the reserve files."""
    name = path.name.lower()
    if "test" in name or name.startswith("reserve_"):
        raise RuntimeError(
            f"Stage 3 must not read {path.name}. Training uses train.csv and dev.csv only."
        )
    return pd.read_csv(path, encoding=encoding)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_datasets(tokenizer, train_df: pd.DataFrame, dev_df: pd.DataFrame,
                   languages: list[str]) -> tuple[Dataset, dict[str, Dataset]]:
    def encode(df: pd.DataFrame) -> Dataset:
        model_inputs = tokenizer(
            [TASK_PREFIX + str(t) for t in df["toxic_input"]],
            max_length=HP["max_source_length"], truncation=True,
        )
        labels = tokenizer(
            text_target=[str(t) for t in df["detox_output"]],
            max_length=HP["max_target_length"], truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return Dataset.from_dict(model_inputs)

    train_ds = encode(train_df)
    # "all" drives model selection; the per-language entries give per-language dev loss.
    eval_ds = {"all": encode(dev_df)}
    for lang in languages:
        eval_ds[lang] = encode(dev_df[dev_df["language"] == lang])
    return train_ds, eval_ds


def train_one_seed(seed: int, cfg: dict, train_df: pd.DataFrame, dev_df: pd.DataFrame,
                   dry_run: bool) -> dict:
    languages = cfg["languages"]
    model_name = cfg["models"]["primary"]
    out_dir = MODEL_DIR / f"seed{seed}"
    log_dir = LOG_DIR / f"seed{seed}"

    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, eval_ds = build_datasets(tokenizer, train_df, dev_df, languages)

    steps_per_epoch = int(np.ceil(
        len(train_ds) / (HP["per_device_train_batch_size"] * HP["gradient_accumulation_steps"])
    ))
    plan = {
        "seed": seed,
        "model": model_name,
        "train_rows": len(train_ds),
        "dev_rows": len(eval_ds["all"]),
        "steps_per_epoch": steps_per_epoch,
        "max_total_steps": steps_per_epoch * HP["num_train_epochs"],
        "warmup_steps": int(steps_per_epoch * HP["num_train_epochs"] * HP["warmup_ratio"]),
        "effective_batch_size": HP["per_device_train_batch_size"] * HP["gradient_accumulation_steps"],
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "log_dir": str(log_dir.relative_to(REPO_ROOT)),
        **HP,
    }
    if dry_run:
        return {"plan": plan, "dry_run": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolved configuration banner, printed before each seed starts training.
    print("  resolved configuration")
    print(f"    model           : {model_name}")
    print(f"    train data      : {(SPLITS_DIR / 'train.csv').relative_to(REPO_ROOT)} "
          f"({len(train_df)} rows)")
    print(f"    dev data        : {(SPLITS_DIR / 'dev.csv').relative_to(REPO_ROOT)} "
          f"({len(dev_df)} rows)")
    print(f"    test data       : NOT READ (guarded)")
    print(f"    reserve files   : NOT READ (guarded)")
    print(f"    seed            : {seed}")
    print(f"    output dir      : {out_dir.relative_to(REPO_ROOT)}")
    print(f"    log dir         : {log_dir.relative_to(REPO_ROOT)}")
    print(f"    steps/epoch     : {steps_per_epoch} (effective batch "
          f"{plan['effective_batch_size']}, ceiling {plan['max_total_steps']} steps)")
    print(f"    precision       : bf16 | lr {HP['learning_rate']} | "
          f"wd {HP['weight_decay']} | patience {HP['early_stopping_patience']}")

    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # transformers v5 removed warmup_ratio; warmup_steps against the full scheduled run
    # (steps_per_epoch * num_train_epochs) is exactly equivalent to the approved 10% ratio.
    warmup_steps = int(steps_per_epoch * HP["num_train_epochs"] * HP["warmup_ratio"])

    args = Seq2SeqTrainingArguments(
        output_dir=str(log_dir / "checkpoints"),
        seed=seed,
        data_seed=seed,
        learning_rate=HP["learning_rate"],
        lr_scheduler_type=HP["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        weight_decay=HP["weight_decay"],
        max_grad_norm=HP["max_grad_norm"],
        num_train_epochs=HP["num_train_epochs"],
        per_device_train_batch_size=HP["per_device_train_batch_size"],
        gradient_accumulation_steps=HP["gradient_accumulation_steps"],
        per_device_eval_batch_size=HP["per_device_eval_batch_size"],
        optim=HP["optimizer"],
        bf16=True,
        fp16=False,  # mT5 overflows to NaN in fp16
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=HP["metric_for_best_model"],
        greater_is_better=False,
        logging_strategy="epoch",
        # Default True silently drops NaN/Inf losses from the log, which would defeat
        # the NaN/Inf audit below. Keep them so a divergent run is visible, not hidden.
        logging_nan_inf_filter=False,
        report_to=[],
        dataloader_num_workers=2,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=HP["early_stopping_patience"])],
    )

    train_result = trainer.train()

    # Persist the selected (best) model, not the last epoch.
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    history = [h for h in trainer.state.log_history if any(k.endswith("_loss") for k in h)]
    final_eval = trainer.evaluate()

    n_epochs_run = max((h.get("epoch", 0) for h in trainer.state.log_history), default=0)
    stopped_early = n_epochs_run < HP["num_train_epochs"]

    # Best epoch: the evaluation with the lowest aggregate dev loss.
    evals = [h for h in history if HP["metric_for_best_model"] in h]
    best_entry = min(evals, key=lambda h: h[HP["metric_for_best_model"]]) if evals else {}
    best_epoch = best_entry.get("epoch")

    # ---- NaN / Inf checks ----
    def _finite(x) -> bool:
        return isinstance(x, (int, float)) and np.isfinite(x)

    nonfinite_train_losses = [
        {"epoch": h.get("epoch"), "loss": h["loss"]}
        for h in trainer.state.log_history
        if "loss" in h and not _finite(h["loss"])
    ]
    nonfinite_dev_losses = {
        k: v for k, v in final_eval.items() if k.endswith("_loss") and not _finite(v)
    }
    bad_params = [
        name for name, p in trainer.model.named_parameters()
        if not torch.isfinite(p).all()
    ]
    nan_checks = {
        "train_loss_all_finite": not nonfinite_train_losses,
        "dev_loss_all_finite": not nonfinite_dev_losses,
        "model_weights_all_finite": not bad_params,
        "nonfinite_train_losses": nonfinite_train_losses,
        "nonfinite_dev_losses": nonfinite_dev_losses,
        "nonfinite_parameter_count": len(bad_params),
        "nonfinite_parameter_examples": bad_params[:5],
        "passed": not (nonfinite_train_losses or nonfinite_dev_losses or bad_params),
    }

    return {
        "plan": plan,
        "dry_run": False,
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "epochs_run": n_epochs_run,
        "stopped_early": stopped_early,
        "hit_epoch_ceiling": not stopped_early,
        "best_epoch": best_epoch,
        "best_checkpoint_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "saved_checkpoint_path": str(out_dir.relative_to(REPO_ROOT)),
        "final_dev_loss": {k: v for k, v in final_eval.items() if k.endswith("_loss")},
        "nan_checks": nan_checks,
        "log_history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print the training plan and exit without training")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="override the seeds from configs/experiment.yaml")
    parsed = parser.parse_args()

    cfg = load_config()
    seeds = parsed.seeds if parsed.seeds else cfg["seeds_multi"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df = guarded_read(SPLITS_DIR / "train.csv")
    dev_df = guarded_read(SPLITS_DIR / "dev.csv")
    print(f"read data/splits/train.csv ({len(train_df)} rows) and "
          f"data/splits/dev.csv ({len(dev_df)} rows); test.csv and reserve files untouched")

    results = {}
    for seed in seeds:
        print(f"\n{'='*70}\nseed {seed}\n{'='*70}")
        res = train_one_seed(seed, cfg, train_df, dev_df, parsed.dry_run)
        results[str(seed)] = res
        if parsed.dry_run:
            for k, v in res["plan"].items():
                print(f"  {k:32s} {v}")
            continue
        print(f"  epochs run        : {res['epochs_run']:.0f} "
              f"(early stop: {res['stopped_early']})")
        print(f"  best epoch        : {res['best_epoch']}")
        print(f"  best dev loss     : {res['best_checkpoint_metric']:.4f}")
        print(f"  NaN/Inf checks    : {'PASS' if res['nan_checks']['passed'] else 'FAIL'}")
        print(f"  checkpoint        : {res['saved_checkpoint_path']}")
        print("  per-language dev loss:")
        for k, v in sorted(res["final_dev_loss"].items()):
            print(f"    {k:24s} {v:.4f}")

    if parsed.dry_run:
        print("\nDRY RUN -- no model was trained, no checkpoint written.")
        return 0

    (LOG_DIR / "dev_loss_history.json").write_text(
        json.dumps({
            "run_id": cfg["run_id"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "task_prefix": TASK_PREFIX,
            "hyperparameters": HP,
            "seeds": results,
        }, indent=2), encoding="utf-8"
    )

    rows = []
    for seed, res in results.items():
        for key, loss in res["final_dev_loss"].items():
            lang = key.replace("eval_", "").replace("_loss", "")
            rows.append({"seed": int(seed), "language": lang, "dev_loss": loss,
                         "epochs_run": res["epochs_run"],
                         "best_epoch": res["best_epoch"],
                         "stopped_early": res["stopped_early"],
                         "nan_checks_passed": res["nan_checks"]["passed"],
                         "checkpoint": res["saved_checkpoint_path"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(LOG_DIR / "training_summary.csv", index=False)

    print("\n=== dev loss by language (mean +/- sd across seeds) ===")
    agg = summary.groupby("language")["dev_loss"].agg(["mean", "std", "min", "max"]).round(4)
    print(agg.to_string())

    print("\n=== per-seed outcome ===")
    for seed, res in results.items():
        print(f"  seed {seed:>5s}: early_stop={str(res['stopped_early']):5s} "
              f"best_epoch={res['best_epoch']!s:>5s} "
              f"epochs_run={res['epochs_run']:.0f} "
              f"best_dev_loss={res['best_checkpoint_metric']:.4f} "
              f"nan_checks={'PASS' if res['nan_checks']['passed'] else 'FAIL'} "
              f"-> {res['saved_checkpoint_path']}")

    failed = [s for s, r in results.items() if not r["nan_checks"]["passed"]]
    if failed:
        print(f"\nWARNING: NaN/Inf detected for seeds {failed}. Details in dev_loss_history.json.")

    ceiling = [s for s, r in results.items() if r["hit_epoch_ceiling"]]
    if ceiling:
        print(f"\nWARNING: seeds {ceiling} reached the {HP['num_train_epochs']}-epoch ceiling "
              f"without early stopping; the budget may be too small.")
    print(f"\nwrote {LOG_DIR.relative_to(REPO_ROOT)}/dev_loss_history.json and training_summary.csv")
    print(f"checkpoints under {MODEL_DIR.relative_to(REPO_ROOT)}/seed<N>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
