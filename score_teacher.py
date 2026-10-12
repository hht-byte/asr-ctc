"""Compare CTC draft token IDs with Qwen3-ASR decoder output IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_pilot import teacher_prefix_metrics


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", required=True, help="best_predictions.jsonl")
    parser.add_argument("--teacher", required=True,
                        help="JSONL: audio, token_ids (content tokens only)")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    drafts = read_rows(Path(args.drafts))
    teachers = read_rows(Path(args.teacher))
    teacher_by_audio = {}
    for row in teachers:
        audio, ids = row.get("audio"), row.get("token_ids")
        if not isinstance(audio, str) or not isinstance(ids, list) or not all(
            isinstance(value, int) and value >= 0 for value in ids
        ):
            raise ValueError("teacher rows need audio and nonnegative integer token_ids")
        if audio in teacher_by_audio:
            raise ValueError(f"duplicate teacher audio: {audio}")
        teacher_by_audio[audio] = ids
    missing = [row["audio"] for row in drafts if row["audio"] not in teacher_by_audio]
    if missing:
        raise ValueError(f"missing teacher tokens for {len(missing)} drafts, first: {missing[0]}")
    metrics = teacher_prefix_metrics(
        [row["draft_token_ids"] for row in drafts],
        [teacher_by_audio[row["audio"]] for row in drafts],
    )
    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
