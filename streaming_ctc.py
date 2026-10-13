"""Two-second cumulative-audio event loading and verifier-aligned metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from ctc_pilot import longest_common_prefix, split_records


def load_stream_manifest(path: str | Path, chunk_samples: int = 32000,
                         min_streams: int = 2) -> list[dict]:
    """Load cumulative teacher transcriptions at every full chunk boundary."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    manifest = Path(path).resolve()
    events: list[dict] = []
    last_end: dict[str, int] = {}
    with manifest.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("audio"), str):
                raise ValueError(f"{manifest}:{number}: expected an audio path")
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{manifest}:{number}: expected cumulative text")
            end = row.get("end_sample")
            if type(end) is not int or end <= 0 or end % chunk_samples:
                raise ValueError(f"{manifest}:{number}: end_sample must be a chunk boundary")
            audio = (manifest.parent / row["audio"]).resolve()
            if not audio.is_file():
                raise ValueError(f"{manifest}:{number}: missing audio: {audio}")
            key = os.path.normcase(str(audio))
            expected = last_end.get(key, 0) + chunk_samples
            if end != expected:
                reason = "first chunk" if key not in last_end else "missing or repeated chunk"
                raise ValueError(f"{manifest}:{number}: {reason}; expected end_sample {expected}")
            last_end[key] = end
            events.append({"audio": str(audio), "end_sample": end,
                           "text": row["text"]})
            if "target_token_ids" in row:
                ids = row["target_token_ids"]
                if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                    raise ValueError(f"{manifest}:{number}: invalid target_token_ids")
                events[-1]["target_token_ids"] = ids
    if len(last_end) < min_streams:
        raise ValueError(f"manifest needs at least {min_streams} recordings")
    return events


def split_stream_records(events: Sequence[dict], dev_fraction: float,
                         seed: int) -> tuple[list[dict], list[dict]]:
    """Split recordings, keeping all their chunk events in the same partition."""
    recordings = list(dict.fromkeys(row["audio"] for row in events))
    train, dev = split_records([{"audio": audio} for audio in recordings],
                               dev_fraction, seed)
    train_paths = {row["audio"] for row in train}
    return ([row for row in events if row["audio"] in train_paths],
            [row for row in events if row["audio"] not in train_paths])


def suffix_after_prefix(full_draft: Sequence[int],
                        stable_prefix: Sequence[int]) -> list[int] | None:
    """Return a safe candidate only when the CTC draft reproduces the prompt prefix."""
    if list(full_draft[:len(stable_prefix)]) != list(stable_prefix):
        return None
    return list(full_draft[len(stable_prefix):])


def streaming_teacher_metrics(
    full_drafts: Sequence[Sequence[int]],
    stable_prefixes: Sequence[Sequence[int]],
    teacher_suffixes: Sequence[Sequence[int]],
    budgets: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, float | int]:
    count = len(full_drafts)
    if not count or len(stable_prefixes) != count or len(teacher_suffixes) != count:
        raise ValueError("streaming evaluation arrays must be nonempty and equally sized")
    candidates = [suffix_after_prefix(draft, prefix)
                  for draft, prefix in zip(full_drafts, stable_prefixes)]
    accepted = [longest_common_prefix(candidate or [], teacher)
                for candidate, teacher in zip(candidates, teacher_suffixes)]
    result: dict[str, float | int] = {
        "events": count,
        "prefix_match_rate": sum(candidate is not None for candidate in candidates) / count,
        "mean_candidate_tokens": sum(len(candidate or []) for candidate in candidates) / count,
        "mean_accepted_tokens": sum(accepted) / count,
    }
    for length in sorted(set(budgets)):
        if type(length) is not int or length < 1:
            raise ValueError("verification budgets must be positive integers")
        result[f"mean_accepted_at_{length}"] = sum(min(value, length) for value in accepted) / count
        result[f"accept_at_least_{length}_rate"] = sum(value >= length for value in accepted) / count
    return result
