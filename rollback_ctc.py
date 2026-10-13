"""Conservative old-draft extension; no target/oracle information enters proposals."""
from __future__ import annotations

from ctc_pilot import longest_common_prefix
from streaming_ctc import suffix_after_prefix

TOKEN_FIELDS = ("prefix_token_ids", "generated_token_ids", "previous_token_ids",
                "old_draft_token_ids", "target_token_ids")


def validate_rollback_traces(rows):
    """Validate complete histories before scoring; input ordering is irrelevant."""
    grouped = {}
    for row in rows:
        if row.get("schema_version") != 2:
            raise ValueError("rollback comparison requires schema_version 2; recollect teacher traces")
        for field in TOKEN_FIELDS:
            ids = row.get(field)
            if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError(f"invalid rollback trace {field}")
        if type(row.get("old_prefix_match")) is not bool:
            raise ValueError("old_prefix_match must be boolean")
        for field in ("chunk_index", "chunk_samples", "end_sample"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"invalid rollback trace {field}")
        grouped.setdefault(row["audio"], []).append(row)
    for audio, history in grouped.items():
        history.sort(key=lambda r: r["end_sample"])
        previous = []
        period = history[0]["chunk_samples"]
        for index, row in enumerate(history, 1):
            if (row["chunk_index"] != index or row["chunk_samples"] != period
                    or row["end_sample"] != index * period):
                raise ValueError(f"{audio}: rollback history must be contiguous from first chunk")
            prefix = row["prefix_token_ids"]
            if index <= 2 and prefix:
                raise ValueError("first two chunks must have empty stable prefix for this protocol")
            if row["previous_token_ids"] != previous:
                raise ValueError(f"{audio}: previous token history mismatch")
            suffix = suffix_after_prefix(previous, prefix)
            if row["old_prefix_match"] != (suffix is not None):
                raise ValueError("old_prefix_match inconsistent with previous output")
            if row["old_draft_token_ids"] != (suffix or []):
                raise ValueError("old draft must be the previous output after the actual rollback prefix")
            previous = prefix + row["generated_token_ids"]
            if row["target_token_ids"] != previous:
                raise ValueError("target token IDs must preserve prefix and actual generated suffix")


def build_hybrid_candidate(full_ctc, trace):
    """First: none. Second: full previous. Later: tail. Append only on exact overlap."""
    if trace["chunk_index"] == 1 or not trace["old_prefix_match"]:
        return []
    old = trace["old_draft_token_ids"]
    extension = suffix_after_prefix(full_ctc, trace["prefix_token_ids"] + old)
    return list(old) + (extension or [])


def rollback_metrics(full_drafts, traces, budgets=(1, 2, 4, 8)):
    """Greedy single-verification content-token proxies, excluding target bonus/EOS.

    Call validate_rollback_traces on the full history before evaluating subsets.
    There is no simulation of subsequent verification rounds or a speedup claim.
    """
    if not traces or len(full_drafts) != len(traces):
        raise ValueError("rollback evaluation arrays must be nonempty and equally sized")
    budgets = sorted(set(budgets))
    if any(type(k) is not int or k < 1 for k in budgets):
        raise ValueError("verification budgets must be positive integers")
    measurements = []
    for ctc, row in zip(full_drafts, traces):
        active = row["chunk_index"] >= 2
        old = row["old_draft_token_ids"] if active else []
        ctc_only = (suffix_after_prefix(ctc, row["prefix_token_ids"]) or []) if active else []
        hybrid = build_hybrid_candidate(ctc, row)
        target = row["generated_token_ids"]
        measurements.append({
            "index": row["chunk_index"], "old_length": len(old),
            "ctc_length": len(ctc_only), "hybrid_length": len(hybrid),
            "appended": len(hybrid) - len(old),
            "reusable": active and row["old_prefix_match"],
            "old": longest_common_prefix(old, target),
            "ctc": longest_common_prefix(ctc_only, target),
            "hybrid": longest_common_prefix(hybrid, target),
        })

    def summarize(items):
        count = len(items)
        denominator = count or 1
        result = {"events": count,
                  "old_prefix_reusable_rate": sum(r["reusable"] for r in items) / denominator,
                  "extension_proposed_rate": sum(r["appended"] > 0 for r in items) / denominator,
                  "mean_appended_tokens": sum(r["appended"] for r in items) / denominator}
        for mode in ("old", "ctc", "hybrid"):
            result[f"{mode}_mean_candidate_tokens"] = sum(r[f"{mode}_length"] for r in items) / denominator
            for k in budgets:
                result[f"{mode}_mean_accepted_at_{k}"] = sum(min(r[mode], k) for r in items) / denominator
        for k in budgets:
            deltas = [min(r["hybrid"], k) - min(r["old"], k) for r in items]
            result[f"mean_delta_accepted_at_{k}"] = sum(deltas) / denominator
            result[f"incremental_gain_rate_at_{k}"] = sum(d > 0 for d in deltas) / denominator
            result[f"old_fills_budget_rate_at_{k}"] = sum(r["old_length"] >= k for r in items) / denominator
        return result

    return {"policy": "no_first_full_second_tail_later_exact_ctc_extension",
            "scope": "single_verification_greedy_content_tokens_no_bonus_or_eos",
            "all": summarize(measurements),
            "after_first": summarize([r for r in measurements if r["index"] >= 2]),
            "by_trigger": {
                "first": summarize([r for r in measurements if r["index"] == 1]),
                "second": summarize([r for r in measurements if r["index"] == 2]),
                "later": summarize([r for r in measurements if r["index"] >= 3])}}
