#!/usr/bin/env python
"""Stage 0 -- Environment check for the multilingual detoxification circuit study.

Records the software/hardware environment, validates that every expected corpus
CSV is present and well formed, captures the git commit the run is pinned to, and
emits the central experiment configuration that all later stages read.

Outputs
-------
configs/experiment.yaml : single source of truth for seeds, split sizes, model ids,
                          hook layers, steering strengths and bootstrap settings.
results/env_report.json : full environment + data validation snapshot.
requirements-lock.txt   : frozen package set for this venv.

Judgment calls documented here (not specified in the brief)
-----------------------------------------------------------
* Run ID format is ``<UTC timestamp>_<short git sha>``. It is derived from the
  commit rather than a random value so a re-run on the same commit is traceable,
  while the timestamp still distinguishes repeated runs.
* ``en_detox_full.csv`` is registered under a separate ``supplementary`` key, never
  in ``languages``. The brief is explicit that it is not a tenth language; keeping
  it out of the language list prevents any later stage from iterating it by accident.
* STA classifier coverage for yo/xh is recorded as ``"to_verify"`` rather than
  guessed here. The brief requires an explicit check against the classifier's own
  supported-language list, which is done in Stage 5 where the model is loaded.
* The xh/yo CSVs carry a UTF-8 BOM, so every reader in this project uses the
  ``utf-8-sig`` encoding. It is set once here as ``io.csv_encoding``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"

CSV_ENCODING = "utf-8-sig"
REQUIRED_COLUMNS = ["language", "toxic_input", "detox_output", "source", "split"]

# The nine study languages. African-language focus is Yoruba and isiXhosa only.
LANGUAGES = {
    "am": "Amharic",
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "xh": "isiXhosa",
    "yo": "Yoruba",
}
AFRICAN_LANGUAGES = ["yo", "xh"]

# Not a language: larger English corpus kept only for optional scaling experiments.
SUPPLEMENTARY = {"en_full": "en_detox_full.csv"}

PACKAGES_OF_INTEREST = [
    "torch", "transformers", "sentence_transformers", "datasets", "accelerate",
    "sentencepiece", "sklearn", "scipy", "numpy", "pandas", "matplotlib",
    "seaborn", "yaml", "sae_lens",
]


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_git_info() -> dict:
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "short_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "uncommitted_files": status.splitlines() if status else [],
    }


def collect_package_versions() -> dict:
    import importlib

    versions = {}
    for name in PACKAGES_OF_INTEREST:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - reported, not raised
            versions[name] = f"MISSING ({type(exc).__name__})"
    return versions


def collect_hardware() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    info["driver_version"] = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    return info


def validate_corpus() -> tuple[dict, list[str]]:
    """Check every expected CSV exists, has the required columns, and is non-empty."""
    report: dict = {"languages": {}, "supplementary": {}}
    problems: list[str] = []

    def _check(path: Path, key: str, bucket: str, expected_lang: str | None) -> None:
        if not path.exists():
            problems.append(f"{key}: missing file {path}")
            report[bucket][key] = {"exists": False}
            return
        df = pd.read_csv(path, encoding=CSV_ENCODING)
        entry = {
            "exists": True,
            "path": str(path.relative_to(REPO_ROOT)),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "columns_ok": list(df.columns) == REQUIRED_COLUMNS,
            "language_values": sorted(df["language"].astype(str).unique().tolist())
            if "language" in df.columns else [],
            "sources": sorted(df["source"].astype(str).unique().tolist())
            if "source" in df.columns else [],
        }
        if not entry["columns_ok"]:
            problems.append(f"{key}: columns {entry['columns']} != {REQUIRED_COLUMNS}")
        if entry["rows"] == 0:
            problems.append(f"{key}: file is empty")
        if expected_lang and entry["language_values"] != [expected_lang]:
            problems.append(
                f"{key}: language column holds {entry['language_values']}, expected ['{expected_lang}']"
            )
        report[bucket][key] = entry

    for code in LANGUAGES:
        _check(DATA_DIR / f"{code}_detox.csv", code, "languages", code)
    for key, fname in SUPPLEMENTARY.items():
        _check(DATA_DIR / fname, key, "supplementary", None)

    # Guard the brief's explicit constraint: no Hausa, no Russian anywhere in the corpus.
    forbidden = sorted(
        {p.name for p in DATA_DIR.glob("*.csv")}
        - {f"{c}_detox.csv" for c in LANGUAGES}
        - set(SUPPLEMENTARY.values())
    )
    report["unexpected_files"] = forbidden
    for name in forbidden:
        if name.startswith(("ha_", "ru_")):
            problems.append(f"forbidden corpus present: {name}")

    return report, problems


def build_config(run_id: str, git_info: dict) -> dict:
    """Central config consumed by Stages 1-17."""
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["commit"],
        "seed": 42,
        "seeds_multi": [42, 1337, 2024],  # >=3 seeds required for probe/SAE/steering results
        "io": {
            "csv_encoding": CSV_ENCODING,
            "data_processed": "data/processed",
            "data_splits": "data/splits",
            "results": "results",
            "models": "models",
        },
        "languages": list(LANGUAGES),
        "language_names": LANGUAGES,
        "african_languages": AFRICAN_LANGUAGES,
        "non_african_languages": [c for c in LANGUAGES if c not in AFRICAN_LANGUAGES],
        "supplementary": {
            "en_full": {
                "path": "data/processed/en_detox_full.csv",
                "note": "Optional supplementary English resource. NOT a separate language; "
                        "excluded from all per-language loops.",
            }
        },
        "splits": {
            "cap_per_language": 178,   # matched to yo/xh corpus size
            "train": 125,
            "dev": 27,
            "test": 26,
            "reserve_pattern": "data/splits/reserve_{lang}.csv",
            "test_policy": "Test split is touched exactly once, for final reported numbers only.",
        },
        "models": {
            "primary": "google/mt5-base",
            "benchmark_reference_only": {
                "id": "s-nlp/mt0-xl-detox-orpo",
                "trained_languages": ["en", "de", "es", "ru", "ar", "hi", "uk", "am", "zh"],
                "note": "Reference baseline only. yo/xh are OUTSIDE its trained set -- any "
                        "yo/xh output must be flagged as unvalidated, not reported as a baseline.",
            },
            "sim": "sentence-transformers/LaBSE",
            "sta": "textdetox/xlmr-large-toxicity-classifier",
        },
        "evaluation": {
            "sta_language_support": {c: ("to_verify" if c in AFRICAN_LANGUAGES else "assumed_supported")
                                     for c in LANGUAGES},
            "sta_verification_note": "Stage 5 must check the classifier's own supported-language "
                                     "list and mark STA 'unavailable / unvalidated' where absent.",
            "bootstrap_resamples": 1000,
            "bootstrap_ci": 0.95,
            "bootstrap_unit": "sentence_pair",
            "joint_score": "J = STA * SIM * FL",
        },
        "activations": {
            "encoder_layers": [4, 6, 8],
            "decoder_layers": [4, 6, 8],
            "probe_sae_data": ["train", "dev"],
            "test_policy": "Test activations collected separately and untouched until final eval.",
        },
        "sae": {
            "primary_sites": [{"side": "encoder", "layer": 6}, {"side": "decoder", "layer": 6}],
            "expansion_factor": 8,
            "activation": "topk",
        },
        "steering": {
            "strengths": [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0],
            "selection_split": "dev",
            "selection_metric": "J",
            "controls": "Every ablation/steering result requires a matched random-component control.",
        },
        "constraints": [
            "No Hausa and no Russian/Russia references anywhere in code, comments, configs or reports.",
            "Pair-level splits, fixed seed, no leakage.",
            "Report negative/failed interventions alongside positive ones.",
            "Shared-circuit claims require cross-language correlation AND a causal effect; "
            "correlation alone is labelled 'associative', not 'causal'.",
            "STA is never used as evidence of grammaticality or meaning preservation.",
            "If steering raises STA, verify SIM did not collapse; flag per language.",
        ],
    }


def main() -> int:
    CONFIG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    git_info = collect_git_info()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{git_info['short_commit']}"

    corpus_report, problems = validate_corpus()

    env_report = {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": collect_package_versions(),
        "hardware": collect_hardware(),
        "corpus_validation": corpus_report,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }

    (RESULTS_DIR / "env_report.json").write_text(
        json.dumps(env_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config = build_config(run_id, git_info)
    with open(CONFIG_DIR / "experiment.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )
    if freeze.returncode == 0:
        (REPO_ROOT / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")

    # ---- console summary ----
    print(f"run_id            : {run_id}")
    print(f"git               : {git_info['short_commit']} on {git_info['branch']} "
          f"(dirty={git_info['dirty']})")
    print(f"python            : {'.'.join(map(str, sys.version_info[:3]))} @ {sys.executable}")
    hw = env_report["hardware"]
    print(f"torch             : {env_report['packages']['torch']} (cuda {hw['torch_cuda_version']})")
    print(f"transformers      : {env_report['packages']['transformers']}")
    print(f"cuda available    : {hw['cuda_available']} | driver {hw['driver_version']}")
    for dev in hw["devices"]:
        print(f"  gpu[{dev['index']}]          : {dev['name']} "
              f"{dev['total_memory_gb']} GB, cc {dev['compute_capability']}")
    print("\ncorpus:")
    for code, entry in corpus_report["languages"].items():
        print(f"  {code} ({LANGUAGES[code]:9s}) rows={entry.get('rows', 0):5d} "
              f"cols_ok={entry.get('columns_ok')} sources={entry.get('sources')}")
    for key, entry in corpus_report["supplementary"].items():
        print(f"  [supp] {key:11s} rows={entry.get('rows', 0):5d} cols_ok={entry.get('columns_ok')}")
    if corpus_report["unexpected_files"]:
        print(f"  unexpected files : {corpus_report['unexpected_files']}")

    print(f"\nstatus            : {env_report['status']}")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
    print("\nwrote configs/experiment.yaml, results/env_report.json, requirements-lock.txt")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
