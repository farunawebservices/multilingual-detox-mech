from pathlib import Path
import re

import pandas as pd

ROOT = Path("/workspace/multilingual-detox-mech")
INFILE = ROOT / "outputs" / "phase2_gpt5_detox" / "afrihate_phase2_candidates_100.csv"
OUTFILE = ROOT / "outputs" / "phase2_gpt5_detox" / "afrihate_phase2_candidates_filtered.csv"

df = pd.read_csv(INFILE).copy()

def normalize(text):
    return " ".join(str(text).casefold().split())

def has_english_dominance(text):
    words = re.findall(r"[a-zA-Z']+", str(text).casefold())
    if not words:
        return False

    english_markers = {
        "the", "and", "you", "your", "are", "this", "that", "with",
        "for", "from", "in", "on", "to", "of", "is", "be", "will",
        "because", "people", "they", "them", "we", "our", "normal",
        "country", "train", "speed", "politicians", "fool", "stupid",
        "bastard", "fuck", "fuckin", "fuckng", "shit", "dead",
    }

    count = sum(word in english_markers for word in words)
    return count >= 4

def exclusion_reason(row):
    text = normalize(row["toxic_input_processing"])
    language = row["language"]

    identity_patterns = {
        "Yorùbá": [
            "omo ale yoruba", "omo igbo", "fulani", "hausa rederede",
            "gambari", "oyinbo", "alawọ funfun", "alawofunfun",
        ],
        "isiXhosa": [
            "umhlophe", "umlungu", "ikwerekwere", "isitabane",
            "nkawu", "qheya", "whitey",
        ],
    }

    violence_patterns = [
        "ma pa", "ma ku", "kú", "kill", "dead", "beat him",
        "shaya", "afe", "die", "bulale", "murder",
    ]

    sexual_patterns = [
        "pipi", "msunu", "incanca", "isifebe", "ihule", "umgodoyi",
        "condom", "balls", "bhakodo", "bhodla",
    ]

    too_short = len(text.split()) < 3
    too_long = len(text.split()) > 35

    if any(pattern in text for pattern in identity_patterns.get(language, [])):
        return "exclude_identity_targeted"

    if any(pattern in text for pattern in violence_patterns):
        return "exclude_violence_or_death"

    if any(pattern in text for pattern in sexual_patterns):
        return "exclude_explicit_sexual_or_gendered"

    if has_english_dominance(text):
        return "exclude_english_dominant"

    if too_short:
        return "exclude_too_short"

    if too_long:
        return "exclude_too_long"

    return "keep"

df["local_filter_decision"] = df.apply(exclusion_reason, axis=1)
kept = df[df["local_filter_decision"] == "keep"].copy()

# Keep the exact source provenance and add a new sequential ID.
kept = kept.reset_index(drop=True)
kept.insert(0, "filtered_row_id", range(1, len(kept) + 1))

kept.to_csv(OUTFILE, index=False, encoding="utf-8")

print("Input rows:", len(df))
print("\nLocal-filter decisions:")
print(df.groupby(["language", "local_filter_decision"]).size())

print("\nRows retained:", len(kept))
print("\nRetained by language and label:")
print(kept.groupby(["language", "label"]).size())

print("\nOutput:", OUTFILE)