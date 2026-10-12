import json
import tempfile
import unittest
from pathlib import Path

from ctc_pilot import (
    build_vocabulary,
    collapse_ctc,
    edit_distance,
    evaluate_predictions,
    load_manifest,
    longest_common_prefix,
    minimum_ctc_frames,
    split_records,
    teacher_prefix_metrics,
)


class PilotTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_audio_and_empty_transcript(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / "a.wav").write_bytes(b"placeholder")
            manifest = root / "data.jsonl"
            manifest.write_text(
                '\n'.join([
                    json.dumps({"audio": "a.wav", "text": "你好"}),
                    json.dumps({"audio": "a.wav", "text": "再次"}),
                ]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(manifest)
            manifest.write_text(json.dumps({"audio": "a.wav", "text": "  "}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_manifest(manifest)

    def test_split_is_deterministic_and_nonoverlapping(self):
        records = [{"audio": str(i), "text": str(i)} for i in range(20)]
        train, dev = split_records(records, dev_fraction=0.2, seed=42)
        self.assertEqual((len(train), len(dev)), (16, 4))
        self.assertEqual((train, dev), split_records(records, dev_fraction=0.2, seed=42))
        self.assertFalse({r["audio"] for r in train} & {r["audio"] for r in dev})

    def test_ctc_vocabulary_and_collapse_preserve_original_token_ids(self):
        vocab = build_vocabulary([[13, 25, 13], [25, 9]])
        self.assertEqual(vocab, [None, 9, 13, 25])
        self.assertEqual(collapse_ctc([0, 2, 2, 0, 2, 3, 3, 0], vocab), [13, 13, 25])

    def test_ctc_feasibility_and_draft_metrics(self):
        self.assertEqual(minimum_ctc_frames([4, 4, 7, 7, 7]), 8)
        self.assertEqual(longest_common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(edit_distance([1, 2, 3], [1, 4, 3, 5]), 2)

    def test_evaluation_counts_token_prefix_and_out_of_vocabulary(self):
        rows = evaluate_predictions(
            predictions=[[10, 11], [10]],
            references=[[10, 11, 12], [10, 13]],
            predicted_texts=["你好", "世"],
            reference_texts=["你好啊", "世界"],
            vocabulary=[None, 10, 11, 12],
        )
        self.assertEqual(rows["token_error_rate"], 2 / 5)
        self.assertEqual(rows["mean_prefix_tokens"], 1.5)
        self.assertEqual(rows["prefix_at_least_2_rate"], 0.5)
        self.assertEqual(rows["dev_oov_token_rate"], 1 / 5)

    def test_teacher_prefix_reports_capped_accepted_draft_tokens(self):
        metrics = teacher_prefix_metrics([[1, 2, 3, 4], [5, 6]],
                                         [[1, 2, 9], [5, 6, 7]])
        self.assertEqual(metrics["mean_prefix_tokens"], 2)
        self.assertEqual(metrics["mean_accepted_at_4"], 2)
        self.assertEqual(metrics["exact_draft_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
