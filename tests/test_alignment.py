"""Offline alignment comparison, with target-independent candidate construction."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare_alignment import compare_rows, propose
from score_teacher import score_rows


def history():
    rows = []
    previous = []
    for i, (prefix, generated) in enumerate(
            [([], [10, 20]), ([], [10, 20, 30]), ([10], [20, 30, 40, 50])], 1):
        rows.append({"audio": "a.wav", "end_sample": i * 32000,
                     "chunk_index": i, "chunk_samples": 32000, "schema_version": 2,
                     "prefix_token_ids": prefix, "generated_token_ids": generated,
                     "previous_token_ids": previous, "old_draft_token_ids": previous[len(prefix):],
                     "old_prefix_match": True, "target_token_ids": prefix + generated})
        previous = prefix + generated
    return rows


def drafts_for(teachers, sequences):
    return [{"audio": t["audio"], "end_sample": t["end_sample"], "draft_token_ids": ids}
            for t, ids in zip(teachers, sequences)]


class AlignmentTests(unittest.TestCase):
    def test_local_tail_recovers_extension_without_matching_stable_prefix(self):
        teachers = history()
        drafts = drafts_for(teachers, [[10, 20], [10, 20, 30], [99, 20, 30, 40, 50]])
        report, events = compare_rows(drafts[::-1], teachers, verify_budget=8)
        full = report["policies"]["full_prefix"]["after_first"]
        local = report["policies"]["rollback_tail_local"]["after_first"]
        self.assertEqual(full["mean_delta_accepted_at_8"], .5)
        self.assertEqual(local["mean_delta_accepted_at_8"], 1.5)
        self.assertEqual(local["alignment_success_rate"], 1)
        self.assertEqual(local["extension_proposed_rate"], 1)
        conditional = local["when_old_fully_accepted"]
        self.assertEqual(conditional["events"], 2)
        self.assertEqual(conditional["mean_appended_accepted_at_8"], 1.5)
        self.assertEqual(report["comparison"]["after_first"]["mean_local_minus_full_accepted_at_8"], 1)
        reference = score_rows(drafts, teachers, require_rollback=True)
        self.assertEqual(full["hybrid_mean_accepted_at_8"],
                         reference["rollback_comparison"]["after_first"]["hybrid_mean_accepted_at_8"])
        self.assertEqual(len(events), 3)

    def test_repeated_tail_is_ambiguous_even_if_only_one_occurrence_has_suffix(self):
        p = propose([20, 30, 40, 20, 30], [10], [20, 30], "rollback_tail_local")
        self.assertEqual(p["reason"], "ambiguous_tail")
        self.assertEqual(p["match_count"], 2)
        self.assertEqual(p["candidate_token_ids"], [20, 30])
        p = propose([99, 20, 30], [10], [20, 30], "rollback_tail_local")
        self.assertTrue(p["alignment_success"])
        self.assertEqual(p["reason"], "aligned_no_extension")

    def test_empty_tail_and_wrong_prefix_do_not_make_local_candidates(self):
        self.assertEqual(propose([10, 20], [10], [], "rollback_tail_local")["reason"],
                         "empty_tail")
        self.assertEqual(propose([99, 20], [10], [20], "full_prefix")["reason"],
                         "prefix_mismatch")

    def test_old_error_prevents_added_acceptance_despite_proposal(self):
        teachers = history()[:2]
        teachers[1]["generated_token_ids"] = [10, 21, 30]
        teachers[1]["target_token_ids"] = [10, 21, 30]
        drafts = drafts_for(teachers, [[10, 20], [10, 20, 30]])
        report, events = compare_rows(drafts, teachers)
        local = report["policies"]["rollback_tail_local"]["after_first"]
        self.assertEqual(local["extension_proposed_rate"], 1)
        self.assertEqual(local["mean_delta_accepted_at_8"], 0)
        self.assertEqual(local["when_old_fully_accepted"]["events"], 0)
        self.assertIsNone(local["when_old_fully_accepted"]["mean_appended_accepted_at_8"])
        self.assertEqual(events[1]["policies"]["rollback_tail_local"]["appended_accepted_tokens"], 0)

    def test_budget_and_unbounded_added_acceptance_are_separate(self):
        teachers = history()[:2]
        report, _ = compare_rows(drafts_for(teachers, [[10, 20], [10, 20, 30]]),
                                 teachers, verify_budget=2)
        row = report["policies"]["rollback_tail_local"]["after_first"]
        self.assertEqual(row["mean_delta_accepted_at_2"], 0)
        self.assertEqual(row["extension_proposed_rate"], 1)
        self.assertEqual(row["extension_within_budget_rate_at_2"], 0)
        self.assertEqual(row["when_old_fully_accepted"]["mean_appended_accepted_tokens"], 1)
        self.assertEqual(row["when_old_fully_accepted"]["budget_room_events_at_2"], 0)

    def test_proposals_do_not_depend_on_current_teacher_tokens(self):
        teachers = history()[:2]
        drafts = drafts_for(teachers, [[10, 20], [10, 20, 30]])
        _, before = compare_rows(drafts, teachers)
        teachers[1].update(generated_token_ids=[99], target_token_ids=[99])
        _, after = compare_rows(drafts, teachers)
        for policy in ("full_prefix", "rollback_tail_local"):
            self.assertEqual(before[1]["policies"][policy]["candidate_token_ids"],
                             after[1]["policies"][policy]["candidate_token_ids"])

    def test_first_and_nonreusable_history_have_no_proposals(self):
        teachers = history()
        teachers[2].update(prefix_token_ids=[777], target_token_ids=[777, 40],
                           generated_token_ids=[40], old_prefix_match=False, old_draft_token_ids=[])
        report, events = compare_rows(
            drafts_for(teachers, [[10, 20], [10, 20, 30], [777, 40]]), teachers)
        for policy in report["policies"]:
            self.assertEqual(events[0]["policies"][policy]["candidate_token_ids"], [])
            self.assertEqual(events[2]["policies"][policy]["candidate_token_ids"], [])
            self.assertEqual(report["policies"][policy]["by_trigger"]["first"]["extension_proposed_rate"], 0)

    def test_cli_and_invalid_history(self):
        teachers = history()
        drafts = drafts_for(teachers, [[10, 20], [10, 20, 30], [99, 20, 30, 40, 50]])
        with self.assertRaises(ValueError):
            compare_rows(drafts[:-1], teachers)
        with self.assertRaises(ValueError):
            compare_rows(drafts, teachers, verify_budget=0)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for name, rows in (("drafts", drafts), ("teacher", teachers)):
                (root / (name + ".jsonl")).write_text(
                    "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            cmd = [sys.executable, "compare_alignment.py", "--drafts", str(root / "drafts.jsonl"),
                   "--teacher", str(root / "teacher.jsonl"), "--output", str(root / "report.json"),
                   "--events-out", str(root / "events.jsonl")]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len((root / "events.jsonl").read_text().splitlines()), 3)
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
