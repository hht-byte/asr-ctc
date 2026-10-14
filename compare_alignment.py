"""Compare full-prefix vs unique rollback-tail alignment using saved dev drafts.

No model, audio encoding, or training is needed. Candidate construction never
receives current teacher output. Scores are single-verification greedy proxies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ctc_pilot import longest_common_prefix
from score_teacher import read_rows, score_rows
from streaming_ctc import suffix_after_prefix

POLICIES = ("full_prefix", "rollback_tail_local")


def propose(ctc, stable_prefix, old, policy):
    """Append only; local alignment uses the entire nonempty old draft as anchor.

    Search all occurrences, including terminal ones; reject ambiguous matches
    without using teacher output, candidate quality, or available suffix length.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown alignment policy: {policy}")
    result = {"candidate_token_ids": list(old), "alignment_success": False,
              "match_count": 0, "match_end": None, "anchor_tokens": 0}
    if policy == "full_prefix":
        anchor = list(stable_prefix) + list(old)
        result["anchor_tokens"] = len(anchor)
        extension = suffix_after_prefix(ctc, anchor)
        if extension is None:
            return {**result, "reason": "prefix_mismatch"}
        end = len(anchor)
        count = 1
    else:
        result["anchor_tokens"] = len(old)
        if not old:
            return {**result, "reason": "empty_tail"}
        ends = [i + len(old) for i in range(len(ctc) - len(old) + 1)
                if list(ctc[i:i + len(old)]) == list(old)]
        count = len(ends)
        if count != 1:
            return {**result, "match_count": count,
                    "reason": "tail_not_found" if not count else "ambiguous_tail"}
        end = ends[0]
        extension = list(ctc[end:])
    return {**result, "candidate_token_ids": list(old) + extension,
            "alignment_success": True, "match_count": count, "match_end": end,
            "reason": "extended" if extension else "aligned_no_extension"}


def summarize(events, policy, budget):
    rows = [event["policies"][policy] for event in events]
    n = len(rows)
    eligible = [r for r in rows if r["eligible"]]
    aligned = sum(r["alignment_success"] for r in rows)
    old_full = [r for r in rows if r["old_fully_accepted"]]
    proposed_full = [r for r in old_full if r["appended_tokens"] > 0]
    key = f"delta_accepted_at_{budget}"

    def mean(items, field):
        return sum(r[field] for r in items) / len(items) if items else None

    recordings = {}
    for event in events:
        recordings.setdefault(event["audio"], []).append(event["policies"][policy][key])
    return {
        "events": n, "eligible_events": len(eligible), "alignment_success_events": aligned,
        "alignment_success_rate": aligned / (n or 1),
        "alignment_success_rate_eligible": aligned / len(eligible) if eligible else None,
        "reason_counts": dict(sorted(Counter(r["reason"] for r in rows).items())),
        "extension_proposed_events": sum(r["appended_tokens"] > 0 for r in rows),
        "extension_proposed_rate": sum(r["appended_tokens"] > 0 for r in rows) / (n or 1),
        f"extension_within_budget_rate_at_{budget}": sum(
            r[f"appended_tokens_at_{budget}"] > 0 for r in rows) / (n or 1),
        "mean_appended_tokens": mean(rows, "appended_tokens"),
        f"mean_appended_tokens_at_{budget}": mean(rows, f"appended_tokens_at_{budget}"),
        f"old_mean_accepted_at_{budget}": mean(rows, f"old_accepted_at_{budget}"),
        f"hybrid_mean_accepted_at_{budget}": mean(rows, f"hybrid_accepted_at_{budget}"),
        f"mean_delta_accepted_at_{budget}": mean(rows, key),
        f"incremental_gain_rate_at_{budget}": sum(r[key] > 0 for r in rows) / (n or 1),
        "recordings": len(recordings),
        f"recordings_with_incremental_gain_at_{budget}": sum(
            any(v > 0 for v in values) for values in recordings.values()),
        f"mean_recording_delta_accepted_at_{budget}": (
            sum(sum(v) / len(v) for v in recordings.values()) / len(recordings)
            if recordings else None),
        "when_old_fully_accepted": {
            "definition": "active, reusable, nonempty old draft fully matches current teacher suffix",
            "events": len(old_full),
            "extension_proposed_events": len(proposed_full),
            f"budget_room_events_at_{budget}": sum(r["old_tokens"] < budget for r in old_full),
            "mean_appended_accepted_tokens": mean(old_full, "appended_accepted_tokens"),
            f"mean_appended_accepted_at_{budget}": mean(old_full, f"appended_accepted_at_{budget}"),
            f"mean_appended_accepted_given_extension_at_{budget}": mean(
                proposed_full, f"appended_accepted_at_{budget}"),
        },
    }


def compare_rows(drafts, teachers, *, verify_budget=8):
    # Reuse the production scorer's ID, coverage, contiguous-history and schema checks.
    score_rows(drafts, teachers, require_rollback=True, verify_budget=verify_budget)
    by_key = {(t["audio"], t["end_sample"]): t for t in teachers}
    events = []
    for draft in sorted(drafts, key=lambda r: (r["audio"], r["end_sample"])):
        trace = by_key[(draft["audio"], draft["end_sample"])]
        active = trace["chunk_index"] >= 2
        eligible = active and trace["old_prefix_match"]
        old = trace["old_draft_token_ids"] if active else []
        event = {"audio": draft["audio"], "end_sample": draft["end_sample"],
                 "chunk_index": trace["chunk_index"],
                 "stable_prefix_token_ids": trace["prefix_token_ids"],
                 "old_draft_token_ids": old, "ctc_token_ids": draft["draft_token_ids"],
                 "policies": {}}
        # All proposals are fixed before consulting the current teacher suffix.
        for policy in POLICIES:
            if eligible:
                proposal = propose(draft["draft_token_ids"], trace["prefix_token_ids"], old, policy)
            else:
                proposal = {"candidate_token_ids": [], "alignment_success": False,
                            "match_count": 0, "match_end": None, "anchor_tokens": 0,
                            "reason": "first_no_draft" if not active else "old_prefix_not_reusable"}
            event["policies"][policy] = proposal
        target = trace["generated_token_ids"]
        old_accepted = longest_common_prefix(old, target)
        for proposal in event["policies"].values():
            hybrid = proposal["candidate_token_ids"]
            accepted = longest_common_prefix(hybrid, target)
            added = max(0, accepted - len(old))
            delta = min(accepted, verify_budget) - min(old_accepted, verify_budget)
            proposal.update({
                "eligible": eligible, "old_tokens": len(old),
                "old_fully_accepted": eligible and bool(old) and old_accepted == len(old),
                "old_accepted_tokens": old_accepted, "hybrid_accepted_tokens": accepted,
                "appended_tokens": len(hybrid) - len(old),
                f"appended_tokens_at_{verify_budget}": max(
                    0, min(len(hybrid), verify_budget) - min(len(old), verify_budget)),
                "appended_accepted_tokens": added,
                f"appended_accepted_at_{verify_budget}": max(0, min(accepted, verify_budget) - len(old)),
                f"old_accepted_at_{verify_budget}": min(old_accepted, verify_budget),
                f"hybrid_accepted_at_{verify_budget}": min(accepted, verify_budget),
                f"delta_accepted_at_{verify_budget}": delta,
            })
        events.append(event)
    groups = {"all": events, "after_first": [r for r in events if r["chunk_index"] >= 2]}
    triggers = {"first": [r for r in events if r["chunk_index"] == 1],
                "second": [r for r in events if r["chunk_index"] == 2],
                "later": [r for r in events if r["chunk_index"] >= 3]}
    report = {"scope": "single_verification_greedy_content_tokens_no_bonus_or_eos",
              "verify_budget": verify_budget,
              "local_alignment_rule": "unique_exact_occurrence_of_entire_nonempty_old_draft",
              "policies": {}, "comparison": {}}
    for policy in POLICIES:
        report["policies"][policy] = {
            **{name: summarize(rows, policy, verify_budget) for name, rows in groups.items()},
            "by_trigger": {name: summarize(rows, policy, verify_budget) for name, rows in triggers.items()},
        }
    for name, rows in {**groups, **triggers}.items():
        diffs = [r["policies"]["rollback_tail_local"][f"hybrid_accepted_at_{verify_budget}"]
                 - r["policies"]["full_prefix"][f"hybrid_accepted_at_{verify_budget}"] for r in rows]
        report["comparison"][name] = {
            "events": len(rows),
            f"mean_local_minus_full_accepted_at_{verify_budget}": (
                sum(diffs) / len(diffs) if diffs else None),
            "local_better_events": sum(d > 0 for d in diffs),
            "local_worse_events": sum(d < 0 for d in diffs),
        }
    return report, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", required=True, help="Existing dev CTC draft JSONL")
    parser.add_argument("--teacher", required=True, help="Matching v2 dev teacher JSONL")
    parser.add_argument("--verify-budget", type=int, default=8)
    parser.add_argument("--output", required=True, help="New comparison JSON file")
    parser.add_argument("--events-out", help="Optional new per-event diagnostic JSONL")
    args = parser.parse_args()
    outputs = [Path(p).resolve() for p in (args.output, args.events_out) if p]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        parser.error("output paths must be distinct new files")
    report, events = compare_rows(read_rows(Path(args.drafts)), read_rows(Path(args.teacher)),
                                  verify_budget=args.verify_budget)
    report["inputs"] = {"drafts": str(Path(args.drafts).resolve()),
                        "teacher": str(Path(args.teacher).resolve())}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
    if args.events_out:
        with Path(args.events_out).open("x", encoding="utf-8") as stream:
            for row in events:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(text)


if __name__ == "__main__":
    main()
