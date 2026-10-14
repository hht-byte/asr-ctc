"""Audit recorded train/dev token coverage without loading a model or audio."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def token_coverage(train_tokens, dev_tokens):
    for rows in (train_tokens, dev_tokens):
        if not rows or any(not isinstance(r, list) or any(type(t) is not int or t < 0 for t in r)
                           for r in rows):
            raise ValueError("nonempty splits of nonnegative integer token ID lists required")
    known = {t for row in train_tokens for t in row}
    if not known:
        raise ValueError("training vocabulary is empty")

    def describe(rows):
        total = sum(map(len, rows))
        oov = sum(t not in known for row in rows for t in row)
        prefixes = []
        for row in rows:
            count = 0
            for token in row[:8]:
                if token not in known:
                    break
                count += 1
            prefixes.append(count)
        nonempty = [row for row in rows if row]
        return {"events": len(rows), "nonempty_events": len(nonempty), "tokens": total,
                "unique_tokens": len({t for r in rows for t in r}),
                "oov_tokens": oov, "oov_token_rate": oov / max(1, total),
                "sequence_representable_rate": sum(all(t in known for t in r) for r in rows) / len(rows),
                "first_token_representable_rate_nonempty": (
                    sum(r[0] in known for r in nonempty) / len(nonempty) if nonempty else None),
                "mean_initial_in_vocab_tokens_at_8": sum(prefixes) / len(rows)}

    return {"vocabulary_source": "train_only", "vocabulary_size_with_blank": len(known) + 1,
            "train": describe(train_tokens), "dev": describe(dev_tokens)}


def read_events(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    seen = set()
    recordings = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("audio"), str):
            raise ValueError("events must contain an audio path")
        end = row.get("end_sample")
        if type(end) is not int or end <= 0:
            raise ValueError("expected streaming events with positive end_sample")
        audio = os.path.normcase(str((path.parent / row["audio"]).resolve()))
        key = (audio, end)
        if key in seen:
            raise ValueError("duplicate event in coverage input")
        seen.add(key)
        recordings.add(audio)
    return rows, recordings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Recorded train_events.jsonl")
    parser.add_argument("--dev", required=True, help="Recorded dev_events.jsonl; not test")
    parser.add_argument("--max-dev-oov", type=float, help="Optional gate in [0,1]")
    parser.add_argument("--output", help="Optional new JSON report file")
    args = parser.parse_args()
    if args.max_dev_oov is not None and not 0 <= args.max_dev_oov <= 1:
        parser.error("max-dev-oov must be in [0,1]")
    train, train_paths = read_events(args.train)
    dev, dev_paths = read_events(args.dev)
    if train_paths & dev_paths:
        raise ValueError("train/dev audio overlap")
    report = token_coverage([r.get("target_token_ids") for r in train],
                            [r.get("target_token_ids") for r in dev])
    report["train_recordings"] = len(train_paths)
    report["dev_recordings"] = len(dev_paths)
    report["gate_passed"] = (args.max_dev_oov is None or
                              report["dev"]["oov_token_rate"] <= args.max_dev_oov)
    report["max_dev_oov"] = args.max_dev_oov
    text = json.dumps(report, indent=2)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    if not report["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
