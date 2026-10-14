import unittest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch


class CoverageTests(unittest.TestCase):
    def test_vocab_only_uses_training_and_counts_occurrences(self):
        from audit_coverage import token_coverage
        result = token_coverage([[10, 20, 10]], [[10, 99, 99], [], [20]])
        self.assertEqual(result["vocabulary_size_with_blank"], 3)
        self.assertEqual(result["dev"]["oov_tokens"], 2)
        self.assertEqual(result["dev"]["oov_token_rate"], 0.5)
        self.assertEqual(result["dev"]["sequence_representable_rate"], 2 / 3)
        self.assertEqual(result["dev"]["mean_initial_in_vocab_tokens_at_8"], 2 / 3)
        self.assertEqual(result["train"]["oov_token_rate"], 0)

    def test_invalid_ids_and_empty_training_vocabulary_rejected(self):
        from audit_coverage import token_coverage
        for train, dev in (([[]], [[1]]), ([[1]], [[True]]), ([[1]], [[-1]])):
            with self.assertRaises(ValueError):
                token_coverage(train, dev)

    def test_cli_gate_and_resolved_audio_overlap(self):
        from audit_coverage import main
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            train, dev = root / "train.jsonl", root / "dev.jsonl"
            train.write_text(json.dumps({"audio": "a.wav", "end_sample": 32000,
                                         "target_token_ids": [10]}) + "\n")
            dev.write_text(json.dumps({"audio": "b.wav", "end_sample": 32000,
                                       "target_token_ids": [99]}) + "\n")
            args = ["audit_coverage.py", "--train", str(train), "--dev", str(dev),
                    "--max-dev-oov", "0.05"]
            with patch.object(sys, "argv", args), self.assertRaises(SystemExit) as failure:
                main()
            self.assertEqual(failure.exception.code, 2)
            dev.write_text(json.dumps({"audio": str(root / "a.wav"), "end_sample": 64000,
                                       "target_token_ids": [10]}) + "\n")
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "overlap"):
                main()
