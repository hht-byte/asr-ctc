"""The deployed policy has no draft at trigger 1, full reuse at 2, tail reuse later."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from collect_teacher import capture_stream_events
from rollback_ctc import build_hybrid_candidate, rollback_metrics, validate_rollback_traces
from train import select_targets, checkpoint_score, select_dev_teacher
from score_teacher import score_rows
from streaming_ctc import load_stream_manifest


def traces():
    result = []
    for index, (prefix, generated, previous) in enumerate([
        ([], [10, 20], []),
        ([], [10, 20, 30], [10, 20]),
        ([10], [20, 30, 40, 50], [10, 20, 30]),
    ], 1):
        result.append({"audio": "a.wav", "end_sample": index * 32000,
                       "chunk_index": index, "chunk_samples": 32000,
                       "schema_version": 2, "prefix_token_ids": prefix,
                       "generated_token_ids": generated,
                       "previous_token_ids": previous,
                       "old_draft_token_ids": previous[len(prefix):],
                       "old_prefix_match": True,
                       "target_token_ids": prefix + generated})
    return result


class RollbackTests(unittest.TestCase):
    def test_score_joins_shuffled_complete_histories_and_custom_budget(self):
        teacher = traces()
        drafts = [{"audio": t["audio"], "end_sample": t["end_sample"],
                   "draft_token_ids": t["target_token_ids"]} for t in teacher]
        report = score_rows(drafts[::-1], teacher, require_rollback=True, verify_budget=3)
        self.assertEqual(report["rollback_comparison"]["after_first"]["mean_delta_accepted_at_3"], 1)
        with self.assertRaisesRegex(ValueError, "coverage"):
            score_rows(drafts[:-1], teacher, require_rollback=True)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            score_rows(drafts, [{k: v for k, v in t.items() if k != "schema_version"}
                               for t in teacher], require_rollback=True)

    def test_automatic_split_filters_whole_teacher_histories_only(self):
        teacher = traces() + [{**row, "audio": "train.wav"} for row in traces()]
        dev = [{"audio": t["audio"], "end_sample": t["end_sample"],
                "draft_token_ids": t["target_token_ids"]} for t in traces()]
        filtered = select_dev_teacher(teacher, dev, auto_split=True)
        report = score_rows(dev, filtered, require_rollback=True)
        self.assertEqual(report["rollback_comparison"]["after_first"]["events"], 2)
        self.assertEqual(len(select_dev_teacher(teacher, dev, auto_split=False)), 6)
        # Filtering must not hide missing events within a selected dev recording.
        with self.assertRaisesRegex(ValueError, "missing teacher"):
            score_rows(dev, select_dev_teacher(teacher[1:], dev, auto_split=True),
                       require_rollback=True)

    def test_manifest_preserves_recorded_ids(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / "a.wav").write_bytes(b"placeholder")
            manifest = root / "events.jsonl"
            manifest.write_text(json.dumps({"audio": "a.wav", "text": "a",
                "end_sample": 32000, "target_token_ids": [777]}), encoding="utf-8")
            self.assertEqual(load_stream_manifest(manifest, min_streams=1)[0]["target_token_ids"], [777])

    def test_first_none_second_full_third_tail_and_incremental_budget(self):
        teacher = traces()
        drafts = [[10, 20, 99], [10, 20, 30, 99], [10, 20, 30, 40, 50]]
        validate_rollback_traces(teacher)
        self.assertEqual([build_hybrid_candidate(d, t) for d, t in zip(drafts, teacher)],
                         [[], [10, 20, 30, 99], [20, 30, 40, 50]])
        report = rollback_metrics(drafts, teacher, budgets=(2, 4))
        self.assertEqual(report["by_trigger"]["first"]["hybrid_mean_accepted_at_4"], 0)
        after = report["after_first"]
        self.assertEqual(after["old_mean_accepted_at_4"], 2)
        self.assertEqual(after["hybrid_mean_accepted_at_4"], 3.5)
        self.assertEqual(after["mean_delta_accepted_at_4"], 1.5)
        self.assertEqual(after["mean_delta_accepted_at_2"], 0)

    def test_ctc_disagreement_falls_back_and_cannot_repair_old_error(self):
        teacher = traces()[1]
        self.assertEqual(build_hybrid_candidate([10, 99, 30], teacher), [10, 20])
        teacher["generated_token_ids"] = [10, 21, 30]
        report = rollback_metrics([[10, 20, 30]], [teacher], budgets=(4,))
        self.assertEqual(report["after_first"]["hybrid_mean_accepted_at_4"], 1)
        self.assertEqual(report["after_first"]["mean_delta_accepted_at_4"], 0)

    def test_token_boundary_mismatch_disables_reuse_and_extension(self):
        teacher = traces()[2]
        teacher.update(prefix_token_ids=[777], target_token_ids=[777, 20, 30, 40, 50],
                       old_prefix_match=False, old_draft_token_ids=[])
        self.assertEqual(build_hybrid_candidate([777, 20, 30, 40], teacher), [])
        history = traces()
        history[2] = teacher
        validate_rollback_traces(history)

    def test_trace_validation_detects_gaps_second_prefix_and_wrong_old(self):
        history = traces()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_rollback_traces(history[1:])
        history = traces()
        history[1]["prefix_token_ids"] = [10]
        with self.assertRaisesRegex(ValueError, "second|first two"):
            validate_rollback_traces(history)
        history = traces()
        history[2]["old_draft_token_ids"] = [30]
        with self.assertRaisesRegex(ValueError, "old draft"):
            validate_rollback_traces(history)

    def test_recorded_target_ids_survive_without_retokenization(self):
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
        rows = [{"text": "same text", "target_token_ids": [901]}]
        tokens, source = select_targets(rows, tokenizer, "auto")
        self.assertEqual((tokens, source), ([[901]], "recorded"))
        with self.assertRaisesRegex(ValueError, "mixed"):
            select_targets(rows + [{"text": "legacy"}], tokenizer, "auto")
        with self.assertRaisesRegex(ValueError, "recorded"):
            select_targets([{"text": "legacy"}], tokenizer, "recorded")

    def test_checkpoint_selection_ignores_first_trigger_ctc_success(self):
        metrics = {"token_error_rate": .5, "mean_accepted_at_8": 100,
                   "prefix_match_rate": 1, "rollback_comparison": {
                       "after_first": {"events": 2, "mean_delta_accepted_at_8": .25,
                                       "hybrid_mean_accepted_at_8": 2.25}}}
        score, mode = checkpoint_score(metrics, "hybrid", 8)
        self.assertEqual(mode, "hybrid")
        self.assertEqual(score, (.25, 2.25, -.5))

    def test_capture_previous_actual_ids_and_protocol(self):
        import numpy as np

        class Tokenizer:
            all_special_ids = [999]
            def encode(self, text, **kwargs):
                return [ord(char) for char in text]

        class Model:
            def generate(self, inputs, **kwargs):
                i = len(inputs[0]["multi_modal_data"]["audio"][0]) // 32000
                generated = {1: [65, 66], 2: [65, 66, 67], 3: [66, 68]}[i]
                return [SimpleNamespace(prompt_token_ids=Tokenizer().encode(inputs[0]["prompt"]),
                        outputs=[SimpleNamespace(token_ids=generated + [999])])]

        class ASR:
            model = Model()
            processor = SimpleNamespace(tokenizer=Tokenizer())
            def init_streaming_state(self, **kwargs):
                return SimpleNamespace(prompt_raw="P:", text="", chunk_id=0)
            def streaming_transcribe(self, chunk, state):
                prefix = "A" if state.chunk_id == 2 else ""
                output = self.model.generate([{"prompt": "P:" + prefix,
                    "multi_modal_data": {"audio": [np.ones((state.chunk_id+1)*32000)]}}])
                state.text = prefix + "".join(chr(x) for x in output[0].outputs[0].token_ids[:-1])
                state.chunk_id += 1
                return state

        events, teacher = capture_stream_events(ASR(), np.ones(96000), 32000,
                                                 "a.wav", "Chinese", 2, 2)
        self.assertEqual([t["old_draft_token_ids"] for t in teacher],
                         [[], [65, 66], [66, 67]])
        self.assertEqual(events[2]["target_token_ids"], [65, 66, 68])
        self.assertEqual(teacher[2]["previous_token_ids"], [65, 66, 67])
        validate_rollback_traces(teacher)


if __name__ == "__main__":
    unittest.main()
