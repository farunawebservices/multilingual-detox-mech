#!/usr/bin/env python
"""Stage 2 -- Balanced, deterministic, leakage-free train/dev/test splits.

Source CSVs under data/processed/ are read only. All normalisation happens in the
derived split data written to data/splits/.

Policy (as specified for this stage)
------------------------------------
1. Drop only exact duplicate (toxic_input, detox_output) rows.
2. Keep distinct detoxification references for a repeated toxic_input.
3. Normalise to Unicode NFC in the derived data only.
4. Group rows sharing a normalised toxic_input; a group lands in exactly one split.
5. Target 175 pair rows per language: 122 train / 27 dev / 26 test.
6. Where grouping makes the exact target impossible, use the nearest valid counts
   and record the actual counts in the manifest.
7. Sample deterministically with seed 42 from languages holding more than 175 pairs.
8. Write train/dev/test CSVs, split_manifest.json, and reserve_<lang>.csv.
9. Assert the leakage and count properties before writing.

Judgment calls documented here (not fixed by the policy above)
--------------------------------------------------------------
* Order of operations is normalise-then-deduplicate. NFC normalisation can reveal
  additional exact duplicates that differ only by encoding form -- Yoruba mixes NFC
  and decomposed forms, so the same visible pair can appear twice under different
  byte sequences. Deduplicating first would leave those behind and violate the
  "no exact duplicate pair remains" assertion. Only leading/trailing whitespace is
  stripped; internal whitespace is left untouched, since collapsing it would alter
  the text more than the policy asks.
* Group allocation fills test first, then dev, then train, walking a seed-42 shuffle
  of the groups and skipping any group that would overshoot the target. With almost
  all groups being singletons this hits the targets exactly; where a multi-row group
  cannot fit, the nearest achievable count is taken and flagged in the manifest.
  Filling the smallest split first protects the evaluation splits from absorbing the
  shortfall, which would otherwise cost statistical power where it matters most.
* Rows dropped as exact duplicates are NOT written to the reserve files. Reserve is
  intended as usable supplementary data for later scaling experiments; seeding it
  with known duplicates would reintroduce the problem downstream. Their counts are
  recorded in the manifest instead.
* Reserve rows are held out at group granularity too, so no reserve row shares a
  toxic input with any split row. Without this, a later scaling experiment that pulls
  from reserve would silently leak test inputs into training.
* The original `split` column of the source CSVs holds provenance labels rather than
  a usable partition, so it is preserved as `orig_split` and `split` is set to the
  partition assigned here. Stable `pair_id` and `group_id` values are added because
  Stage 6 keys activations by pair, and `source_row_index` keeps every derived row
  traceable to the untouched source file.
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

TARGETS = {"train": 122, "dev": 27, "test": 26}
TARGET_TOTAL = sum(TARGETS.values())  # 175
FILL_ORDER = ["test", "dev", "train"]  # smallest first; protects evaluation splits

OUT_COLUMNS = [
    "pair_id", "group_id", "language", "toxic_input", "detox_output",
    "source", "orig_split", "split", "source_row_index",
]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def prepare_language(path: Path, lang: str, encoding: str) -> tuple[pd.DataFrame, dict]:
    """Read, NFC-normalise, drop exact duplicate pairs, and assign group ids."""
    raw = pd.read_csv(path, encoding=encoding)
    n_source = len(raw)

    df = pd.DataFrame({
        "language": lang,
        "toxic_input": raw["toxic_input"].map(nfc),
        "detox_output": raw["detox_output"].map(nfc),
        "source": raw["source"].astype(str),
        "orig_split": raw["split"].astype(str),
        "source_row_index": raw.index.astype(int),
    })

    n_non_nfc = int(sum(
        1 for a, b in zip(raw["toxic_input"].astype(str), df["toxic_input"])
        if a.strip() != b
    ) + sum(
        1 for a, b in zip(raw["detox_output"].astype(str), df["detox_output"])
        if a.strip() != b
    ))

    # Policy 1: drop exact duplicate pairs (post-normalisation, see docstring).
    dup_mask = df.duplicated(subset=["toxic_input", "detox_output"], keep="first")
    n_dropped = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)

    # Policy 4: group by normalised toxic input.
    group_index = {t: i for i, t in enumerate(sorted(df["toxic_input"].unique()))}
    df["group_id"] = df["toxic_input"].map(lambda t: f"{lang}-g{group_index[t]:04d}")
    df["pair_id"] = [f"{lang}-{i:04d}" for i in range(len(df))]

    stats = {
        "source_rows": n_source,
        "strings_renormalised_to_nfc": n_non_nfc,
        "exact_duplicate_pairs_dropped": n_dropped,
        "rows_after_dedup": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "n_multirow_groups": int((df.groupby("group_id").size() > 1).sum()),
        "max_group_size": int(df.groupby("group_id").size().max()),
    }
    return df, stats


def allocate_groups(df: pd.DataFrame, seed: int, lang: str) -> dict[str, str]:
    """Assign each group_id to train/dev/test/reserve, keeping groups intact."""
    sizes = df.groupby("group_id").size().to_dict()
    group_ids = sorted(sizes)                      # deterministic starting order
    rng = random.Random(f"{seed}-{lang}")          # stable across runs and platforms
    rng.shuffle(group_ids)

    assignment: dict[str, str] = {}
    remaining = dict(TARGETS)
    for split in FILL_ORDER:
        for gid in group_ids:
            if remaining[split] == 0:
                break
            if gid in assignment:
                continue
            if sizes[gid] <= remaining[split]:
                assignment[gid] = split
                remaining[split] -= sizes[gid]

    for gid in group_ids:                          # policy 8: everything else is reserve
        assignment.setdefault(gid, "reserve")
    return assignment


def run_assertions(splits: dict[str, pd.DataFrame], reserve: dict[str, pd.DataFrame],
                   manifest: dict, languages: list[str]) -> list[str]:
    """Policy 9. Raises AssertionError on violation; returns the list of checks passed."""
    checks: list[str] = []
    combined = pd.concat(list(splits.values()), ignore_index=True)

    # (a) no normalised toxic_input occurs in more than one split
    per_input_splits = combined.groupby("toxic_input")["split"].nunique()
    straddling = per_input_splits[per_input_splits > 1]
    assert straddling.empty, (
        f"{len(straddling)} toxic inputs occur in more than one split, e.g. "
        f"{straddling.index[:3].tolist()}"
    )
    checks.append("no normalized toxic_input occurs in more than one split")

    # (a') the same, extended to the reserve files
    res_all = pd.concat(list(reserve.values()), ignore_index=True) if reserve else pd.DataFrame(
        columns=["toxic_input"]
    )
    if len(res_all):
        overlap = set(res_all["toxic_input"]) & set(combined["toxic_input"])
        assert not overlap, f"{len(overlap)} toxic inputs appear in both reserve and a split"
    checks.append("no toxic_input shared between reserve and any split")

    # (b) no exact duplicate pair remains
    dups = combined.duplicated(subset=["toxic_input", "detox_output"]).sum()
    assert dups == 0, f"{dups} exact duplicate pairs remain across the splits"
    checks.append("no exact duplicate (toxic_input, detox_output) pair remains")

    # (c) every language has its reported count
    for lang in languages:
        for split, df in splits.items():
            actual = int((df["language"] == lang).sum())
            reported = manifest["per_language"][lang]["counts"][split]
            assert actual == reported, (
                f"{lang}/{split}: {actual} rows on disk vs {reported} reported"
            )
    checks.append("every language's on-disk count matches the manifest")

    # (d) test rows are not used for training or development
    test_ids = set(splits["test"]["pair_id"])
    for split in ("train", "dev"):
        clash = test_ids & set(splits[split]["pair_id"])
        assert not clash, f"{len(clash)} test pair_ids also appear in {split}"
    test_pairs = set(zip(splits["test"]["toxic_input"], splits["test"]["detox_output"]))
    for split in ("train", "dev"):
        other = set(zip(splits[split]["toxic_input"], splits[split]["detox_output"]))
        clash = test_pairs & other
        assert not clash, f"{len(clash)} test pairs reappear verbatim in {split}"
    checks.append("no test pair_id or test pair content appears in train or dev")

    # (e) pair_id uniqueness across everything written
    everything = pd.concat([combined] + list(reserve.values()), ignore_index=True)
    assert everything["pair_id"].is_unique, "pair_id collision across splits/reserve"
    checks.append("pair_id is unique across all written files")

    return checks


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    encoding = cfg["io"]["csv_encoding"]
    data_dir = REPO_ROOT / cfg["io"]["data_processed"]
    languages = cfg["languages"]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    per_language: dict[str, dict] = {}
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "dev": [], "test": []}
    reserve: dict[str, pd.DataFrame] = {}

    for lang in languages:
        df, stats = prepare_language(data_dir / f"{lang}_detox.csv", lang, encoding)
        assignment = allocate_groups(df, seed, lang)
        df["split"] = df["group_id"].map(assignment)

        counts = {s: int((df["split"] == s).sum()) for s in ("train", "dev", "test")}
        n_reserve = int((df["split"] == "reserve").sum())

        for s in ("train", "dev", "test"):
            parts[s].append(df.loc[df["split"] == s, OUT_COLUMNS])

        res = df.loc[df["split"] == "reserve", OUT_COLUMNS]
        reserve[lang] = res
        res.to_csv(SPLITS_DIR / f"reserve_{lang}.csv", index=False, encoding="utf-8")

        total = sum(counts.values())
        per_language[lang] = {
            **stats,
            "counts": counts,
            "reserve": n_reserve,
            "split_total": total,
            "target_total": TARGET_TOTAL,
            "target_counts": dict(TARGETS),
            "exact_target_met": counts == TARGETS,
            "shortfall": {s: TARGETS[s] - counts[s] for s in TARGETS if counts[s] != TARGETS[s]},
        }
        flag = "" if counts == TARGETS else "  <-- nearest valid counts"
        print(f"  {lang}: source={stats['source_rows']:5d} dedup={stats['rows_after_dedup']:4d} "
              f"groups={stats['n_groups']:4d} -> train={counts['train']:3d} dev={counts['dev']:3d} "
              f"test={counts['test']:3d} reserve={n_reserve:4d}{flag}")

    splits = {s: pd.concat(parts[s], ignore_index=True) for s in ("train", "dev", "test")}

    manifest = {
        "run_id": cfg["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": cfg["git_commit"],
        "seed": seed,
        "policy": {
            "dedup": "exact (toxic_input, detox_output) duplicates removed after NFC normalisation",
            "multi_reference": "distinct references for a repeated toxic_input are preserved",
            "normalisation": "Unicode NFC applied to derived split data only; source CSVs untouched",
            "grouping": "rows sharing a normalised toxic_input are kept in one split",
            "target_per_language": {"total": TARGET_TOTAL, **TARGETS},
            "sampling": f"deterministic, seed {seed}, per-language RNG stream",
            "fill_order": FILL_ORDER,
            "reserve": "unused groups, held out whole so no toxic input is shared with a split",
        },
        "per_language": per_language,
        "totals": {s: int(len(df)) for s, df in splits.items()},
    }

    checks = run_assertions(splits, reserve, manifest, languages)
    manifest["assertions_passed"] = checks

    for s, df in splits.items():
        df.to_csv(SPLITS_DIR / f"{s}.csv", index=False, encoding="utf-8")
    (SPLITS_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== totals ===")
    for s, df in splits.items():
        print(f"  {s:5s}: {len(df):5d} rows, {df['language'].nunique()} languages")
    print(f"  reserve: {sum(len(r) for r in reserve.values()):5d} rows")

    print("\n=== leakage / integrity assertions ===")
    for c in checks:
        print(f"  PASS  {c}")

    off_target = [l for l, v in per_language.items() if not v["exact_target_met"]]
    print(f"\nlanguages at exact 122/27/26 : {len(languages) - len(off_target)}/{len(languages)}")
    if off_target:
        for lang in off_target:
            print(f"  {lang}: {per_language[lang]['counts']} "
                  f"(shortfall {per_language[lang]['shortfall']})")
    print("\nwrote data/splits/{train,dev,test}.csv, split_manifest.json, reserve_<lang>.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
