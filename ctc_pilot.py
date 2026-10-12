"""Pure data and metric helpers for the Qwen3-ASR CTC feasibility pilot."""

from __future__ import annotations

import json
import os
import random
import unicodedata
from pathlib import Path
from typing import Any, Sequence


def load_manifest(path: str | Path, min_records: int = 2) -> list[dict[str, str]]:
    manifest = Path(path).resolve()
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
                raise ValueError(f"{manifest}:{number}: expected an audio path")
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"{manifest}:{number}: empty transcript")
            audio = (manifest.parent / item["audio"]).resolve()
            if not audio.is_file():
                raise ValueError(f"{manifest}:{number}: missing audio: {audio}")
            key = os.path.normcase(str(audio))
            if key in seen:
                raise ValueError(f"{manifest}:{number}: duplicate audio: {audio}")
            seen.add(key)
            records.append({"audio": str(audio), "text": item["text"].strip()})
    if len(records) < min_records:
        raise ValueError(f"manifest needs at least {min_records} clips")
    return records


def split_records(
    records: Sequence[dict[str, str]], dev_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(records) < 2 or not 0 < dev_fraction < 1:
        raise ValueError("need at least two records and 0 < dev_fraction < 1")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    dev_count = min(len(records) - 1, max(1, round(len(records) * dev_fraction)))
    return ([records[i] for i in order[dev_count:]],
            [records[i] for i in order[:dev_count]])


def build_vocabulary(token_sequences: Sequence[Sequence[int]]) -> list[int | None]:
    ids = sorted({int(token) for tokens in token_sequences for token in tokens})
    if not ids or ids[0] < 0:
        raise ValueError("training transcripts must contain nonnegative tokenizer IDs")
    return [None, *ids]  # CTC blank is local ID 0, never a Qwen tokenizer ID.


def collapse_ctc(local_ids: Sequence[int], vocabulary: Sequence[int | None]) -> list[int]:
    output: list[int] = []
    previous: int | None = None
    for local_id in local_ids:
        if local_id < 0 or local_id >= len(vocabulary):
            raise ValueError(f"CTC ID {local_id} is outside the pilot vocabulary")
        if local_id != previous and local_id != 0:
            token = vocabulary[local_id]
            assert token is not None
            output.append(token)
        previous = local_id
    return output


def minimum_ctc_frames(tokens: Sequence[int]) -> int:
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def longest_common_prefix(left: Sequence[Any], right: Sequence[Any]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for col, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (a != b)))
        previous = current
    return previous[-1]


def _characters(text: str) -> list[str]:
    return list("".join(unicodedata.normalize("NFKC", text).casefold().split()))


def evaluate_predictions(
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    predicted_texts: Sequence[str],
    reference_texts: Sequence[str],
    vocabulary: Sequence[int | None],
) -> dict[str, float | int]:
    count = len(references)
    if not count or any(len(rows) != count for rows in
                        (predictions, predicted_texts, reference_texts)):
        raise ValueError("evaluation arrays must be nonempty and have equal lengths")
    known = set(vocabulary[1:])
    reference_tokens = sum(map(len, references))
    reference_chars = sum(len(_characters(text)) for text in reference_texts)
    prefixes = [longest_common_prefix(p, r) for p, r in zip(predictions, references)]
    metrics: dict[str, float | int] = {
        "samples": count,
        "token_error_rate": sum(edit_distance(p, r) for p, r in zip(predictions, references))
        / max(1, reference_tokens),
        "char_error_rate": sum(edit_distance(_characters(p), _characters(r))
                               for p, r in zip(predicted_texts, reference_texts))
        / max(1, reference_chars),
        "mean_prefix_tokens": sum(prefixes) / count,
        "mean_draft_tokens": sum(map(len, predictions)) / count,
        "dev_oov_token_rate": sum(t not in known for row in references for t in row)
        / max(1, reference_tokens),
    }
    for length in (1, 2, 4, 8):
        metrics[f"prefix_at_least_{length}_rate"] = sum(p >= length for p in prefixes) / count
    return metrics


def teacher_prefix_metrics(
    drafts: Sequence[Sequence[int]], teacher_tokens: Sequence[Sequence[int]]
) -> dict[str, float | int]:
    if not drafts or len(drafts) != len(teacher_tokens):
        raise ValueError("draft and teacher arrays must be nonempty and equally sized")
    prefixes = [longest_common_prefix(draft, teacher)
                for draft, teacher in zip(drafts, teacher_tokens)]
    count = len(drafts)
    result: dict[str, float | int] = {
        "samples": count,
        "mean_prefix_tokens": sum(prefixes) / count,
        "exact_draft_rate": sum(bool(draft) and prefix == len(draft)
                                for prefix, draft in zip(prefixes, drafts)) / count,
    }
    for length in (1, 2, 4, 8):
        result[f"prefix_at_least_{length}_rate"] = sum(p >= length for p in prefixes) / count
        result[f"mean_accepted_at_{length}"] = sum(min(p, length) for p in prefixes) / count
    return result
