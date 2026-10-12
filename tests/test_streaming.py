"""Streaming events must use only audio available at each two-second trigger."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from streaming_ctc import (load_stream_manifest, split_stream_records,
                           streaming_teacher_metrics, suffix_after_prefix)
from score_teacher import score_rows
from stream_draft import full_chunk_ends, generate_drafts
from collect_teacher import capture_stream_events


class StreamingTests(unittest.TestCase):
    def test_train_help_does_not_load_model_dependencies(self):
        result = subprocess.run([sys.executable, "train.py", "--help"],
                                capture_output=True, text=True, cwd=Path.cwd())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--streaming", result.stdout)

    def test_manifest_keeps_all_events_of_one_recording_in_one_split(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            rows = []
            for name in ("a", "b", "c", "d"):
                (root / f"{name}.wav").write_bytes(b"fake")
                for end in (32000, 64000):
                    rows.append({"audio": f"{name}.wav", "end_sample": end,
                                 "text": name})
            manifest = root / "events.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows),
                                encoding="utf-8")
            events = load_stream_manifest(manifest, chunk_samples=32000)
            train, dev = split_stream_records(events, 0.25, 42)
            self.assertEqual((len(train), len(dev)), (6, 2))
            self.assertFalse({row["audio"] for row in train} &
                             {row["audio"] for row in dev})
            self.assertEqual([row["end_sample"] for row in dev], [32000, 64000])

    def test_manifest_rejects_missing_chunk_and_future_audio(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / "a.wav").write_bytes(b"fake")
            manifest = root / "events.jsonl"
            manifest.write_text(json.dumps({"audio": "a.wav", "end_sample": 64000,
                                            "text": "hello"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "first chunk"):
                load_stream_manifest(manifest, 32000)
            manifest.write_text(json.dumps({"audio": "a.wav", "end_sample": 32001,
                                            "text": "hello"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chunk boundary"):
                load_stream_manifest(manifest, 32000)

    def test_teacher_scoring_rejects_draft_when_stable_prefix_diverges(self):
        self.assertEqual(suffix_after_prefix([10, 20, 30], [10, 20]), [30])
        self.assertIsNone(suffix_after_prefix([10, 99, 30], [10, 20]))
        report = streaming_teacher_metrics(
            [[10, 20, 30, 40], [10, 99, 50]],
            [[10, 20], [10, 20]],
            [[30, 41], [50]],
        )
        self.assertEqual(report["events"], 2)
        self.assertEqual(report["prefix_match_rate"], 0.5)
        self.assertEqual(report["mean_accepted_at_2"], 0.5)
        self.assertEqual(report["mean_candidate_tokens"], 1)

    def test_teacher_rows_join_by_audio_and_chunk_boundary(self):
        drafts = [{"audio": "a.wav", "end_sample": 32000,
                   "draft_token_ids": [1, 2], "encoder_head_ms": 100.0,
                   "chunk_seconds": 2.0},
                  {"audio": "a.wav", "end_sample": 64000,
                   "draft_token_ids": [1, 2, 3], "encoder_head_ms": 2500.0,
                   "chunk_seconds": 2.0}]
        teachers = [{"audio": "a.wav", "end_sample": 64000,
                     "prefix_token_ids": [1, 2], "generated_token_ids": [3, 4]},
                    {"audio": "a.wav", "end_sample": 32000,
                     "prefix_token_ids": [], "generated_token_ids": [1, 2]}]
        report = score_rows(drafts, teachers)
        self.assertEqual(report["events"], 2)
        self.assertEqual(report["mean_accepted_tokens"], 1.5)
        self.assertEqual(report["mean_encoder_head_ms"], 1300.0)
        self.assertEqual(report["encoder_head_over_chunk_rate"], 0.5)
        self.assertEqual(report["by_trigger"]["first"]["events"], 1)
        self.assertEqual(report["by_trigger"]["second"]["events"], 1)
        with self.assertRaisesRegex(ValueError, "missing teacher"):
            score_rows(drafts, teachers[:1])

    def test_stream_draft_reencodes_each_cumulative_prefix(self):
        import numpy as np
        import torch
        from unittest.mock import patch

        self.assertEqual(full_chunk_ends(70000, 32000), [32000, 64000])
        seen = []

        def fake_encode(_, __, waveform, ___, backend):
            seen.append(len(waveform))
            return torch.ones(3, 2)

        class Head:
            def __call__(self, states):
                return torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

        with patch("stream_draft.encode_waveform", side_effect=fake_encode):
            rows = generate_drafts(np.ones(70000), [32000, 64000], Head(),
                                   object(), object(), "cpu", "hf", [None, 7],
                                   "a.wav")
        self.assertEqual(seen, [32000, 64000])
        self.assertEqual([row["end_sample"] for row in rows], [32000, 64000])
        self.assertEqual([row["draft_token_ids"] for row in rows], [[7], [7]])
        self.assertTrue(all(row["encoder_head_ms"] >= 0 for row in rows))
        self.assertTrue(all(row["head_ms"] >= 0 for row in rows))

    def test_teacher_capture_records_actual_generated_suffix_per_chunk(self):
        import numpy as np
        from types import SimpleNamespace

        class Tokenizer:
            all_special_ids = [99]

            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

        class Model:
            def generate(self, inputs, **kwargs):
                end = len(inputs[0]["multi_modal_data"]["audio"][0])
                token = 65 if end == 32000 else 66
                output = SimpleNamespace(token_ids=[token, 99], text=chr(token))
                return [SimpleNamespace(outputs=[output])]

        class ASR:
            model = Model()
            processor = SimpleNamespace(tokenizer=Tokenizer())

            def init_streaming_state(self, **kwargs):
                return SimpleNamespace(prompt_raw="P:", text="", chunk_id=0)

            def streaming_transcribe(self, chunk, state):
                prefix = "" if state.chunk_id == 0 else "A"
                self.model.generate([{"prompt": state.prompt_raw + prefix,
                                      "multi_modal_data": {"audio": [np.ones(
                                          (state.chunk_id + 1) * 32000)]}}])
                state.text += "A" if state.chunk_id == 0 else "B"
                state.chunk_id += 1
                return state

        events, teachers = capture_stream_events(ASR(), np.ones(70000), 32000,
                                                 "a.wav", "Chinese", 2, 5)
        self.assertEqual([row["end_sample"] for row in events], [32000, 64000])
        self.assertEqual([row["text"] for row in events], ["A", "AB"])
        self.assertEqual(teachers[1]["prefix_token_ids"], [65])
        self.assertEqual(teachers[1]["generated_token_ids"], [66])


if __name__ == "__main__":
    unittest.main()
