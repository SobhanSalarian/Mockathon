#!/usr/bin/env python3
"""Assemble the final domain-ft SFT dataset.

Combines:
  - data/sentiment_distilled.jsonl  (AFR sentiment pairs distilled from the brain; all)
  - Technical-assessment pairs       (from the general set; sampled)  -> sentiment_assess technical mode
  - RBA-decision pairs               (from the general set; all)      -> macro robustness

Drops the low-value templated 'cautious' sentiment and portfolio samples.
Writes {"input","output"} JSONL (the schema train_1node.py expects) as
data/train.jsonl / val.jsonl / test.jsonl (80/10/10).
"""
import argparse, glob, json, os, random
from collections import Counter


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled", default="data/sentiment_distilled.jsonl")
    ap.add_argument("--general_glob", default="data/train.jsonl,data/val.jsonl,data/test.jsonl",
                    help="comma-separated existing files to mine technical/RBA pairs from")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_technical", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)

    distilled = load(a.distilled)
    if not distilled:
        raise SystemExit(f"No distilled data at {a.distilled} — run 02b_distill_sentiment.py first.")
    sentiment = [{"input": r["input"], "output": r["output"]} for r in distilled]

    # mine technical + RBA pairs from the general set (read BEFORE we overwrite the files)
    general = []
    for p in a.general_glob.split(","):
        general.extend(load(p.strip()))
    technical = [{"input": r["input"], "output": r["output"]} for r in general
                 if r.get("output", "").startswith("Technical Assessment")]
    rba = [{"input": r["input"], "output": r["output"]} for r in general
           if r.get("output", "").startswith("RBA Decision Analysis")]

    random.shuffle(technical)
    technical = technical[:a.n_technical]
    # dedupe RBA by input (the general set repeats them across splits)
    seen, rba_u = set(), []
    for r in rba:
        if r["input"] not in seen:
            seen.add(r["input"]); rba_u.append(r)

    combined = sentiment + technical + rba_u
    random.shuffle(combined)
    n = len(combined)
    splits = {
        "train": combined[:int(n * 0.8)],
        "val":   combined[int(n * 0.8):int(n * 0.9)],
        "test":  combined[int(n * 0.9):],
    }
    os.makedirs(a.out_dir, exist_ok=True)
    for name, data in splits.items():
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    print(f"composition: sentiment={len(sentiment)} technical={len(technical)} rba={len(rba_u)}")
    print(f"total={n}  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    # sentiment balance in the distilled portion
    print(f"distilled sentiment balance: {dict(Counter(r.get('sentiment','?') for r in distilled))}")


if __name__ == "__main__":
    main()
