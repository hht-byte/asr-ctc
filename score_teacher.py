"""Compare CTC draft token IDs with Qwen3-ASR decoder output IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_pilot import teacher_prefix_metrics
from streaming_ctc import streaming_teacher_metrics


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_rows(drafts: list[dict], teachers: list[dict]) -> dict:
    if not drafts or not teachers:
        raise ValueError("draft and teacher files must be nonempty")
    streaming = "end_sample" in drafts[0]
    if any(("end_sample" in row) != streaming for row in drafts + teachers):
        raise ValueError("draft and teacher rows must use the same streaming schema")
    teacher_by_event = {}
    for row in teachers:
        audio = row.get("audio")
        end = row.get("end_sample") if streaming else None
        key = (audio, end)
        if not isinstance(audio, str) or (streaming and (type(end) is not int or end <= 0)):
            raise ValueError("teacher rows need audio and a valid end_sample")
        fields = ("prefix_token_ids", "generated_token_ids") if streaming else ("token_ids",)
        for field in fields:
            ids = row.get(field)
            if not isinstance(ids, list) or not all(type(value) is int and value >= 0
                                                    for value in ids):
                raise ValueError(f"teacher rows need nonnegative integer {field}")
        if key in teacher_by_event:
            raise ValueError(f"duplicate teacher event: {key}")
        teacher_by_event[key] = row
    aligned = []
    seen = set()
    for row in drafts:
        key = (row.get("audio"), row.get("end_sample") if streaming else None)
        if key in seen:
            raise ValueError(f"duplicate draft event: {key}")
        seen.add(key)
        if key not in teacher_by_event:
            raise ValueError(f"missing teacher tokens for draft event: {key}")
        aligned.append(teacher_by_event[key])
    if streaming:
        full_drafts = [row["draft_token_ids"] for row in drafts]
        prefixes = [row["prefix_token_ids"] for row in aligned]
        suffixes = [row["generated_token_ids"] for row in aligned]
        metrics = streaming_teacher_metrics(
            full_drafts, prefixes, suffixes,
        )
        ends_by_audio = {}
        for row in drafts:
            ends_by_audio.setdefault(row["audio"], []).append(row["end_sample"])
        position = {(audio, end): number for audio, ends in ends_by_audio.items()
                    for number, end in enumerate(sorted(ends), 1)}
        groups = {"first": [], "second": [], "later": []}
        for index, row in enumerate(drafts):
            number = position[(row["audio"], row["end_sample"])]
            groups["first" if number == 1 else "second" if number == 2
                   else "later"].append(index)
        metrics["by_trigger"] = {
            name: streaming_teacher_metrics(
                [full_drafts[i] for i in indices], [prefixes[i] for i in indices],
                [suffixes[i] for i in indices])
            for name, indices in groups.items() if indices
        }
        latency_rows = ["encoder_head_ms" in row for row in drafts]
        if any(latency_rows) and not all(latency_rows):
            raise ValueError("encoder_head_ms must be present in every streaming draft row")
        if all(latency_rows):
            times = [row["encoder_head_ms"] for row in drafts]
            periods = [row.get("chunk_seconds") for row in drafts]
            if any(type(t) not in (int, float) or t < 0 for t in times) or any(
                type(p) not in (int, float) or p <= 0 for p in periods
            ):
                raise ValueError("latency rows need nonnegative time and positive chunk_seconds")
            metrics["mean_encoder_head_ms"] = sum(times) / len(times)
            metrics["encoder_head_over_chunk_rate"] = sum(
                t > p * 1000 for t, p in zip(times, periods)) / len(times)
        return metrics
    return teacher_prefix_metrics(
        [row["draft_token_ids"] for row in drafts],
        [row["token_ids"] for row in aligned],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", required=True, help="best_predictions.jsonl")
    parser.add_argument("--teacher", required=True,
                        help="JSONL: audio, token_ids (content tokens only)")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    drafts = read_rows(Path(args.drafts))
    teachers = read_rows(Path(args.teacher))
    metrics = score_rows(drafts, teachers)
    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
