"""Exercise the real CTC training loop with synthetic frozen encoder features."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class TrainingSmokeTests(unittest.TestCase):
    def test_distributed_sampler_does_not_duplicate_small_dataset(self):
        import torch
        from train import make_loader

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for index in range(10):
                torch.save({"features": torch.ones(4, 2) * index},
                           root / f"{index:06d}.pt")
            used = []
            for rank in range(8):
                loader, sampler = make_loader(
                    list(range(10)), [[1]] * 10, root, [None, 1],
                    batch_size=2, rank=rank, world=8, shuffle=True, seed=42
                )
                sampler.set_epoch(1)
                for features, *_ in loader:
                    used.extend(int(row[0, 0]) for row in features)
            self.assertEqual(len(used), 8)
            self.assertEqual(len(set(used)), 8)

    def test_both_encoder_input_contracts(self):
        import numpy as np
        import torch
        from model_io import encode_audio

        soundfile = types.ModuleType("soundfile")
        soundfile.read = lambda *args, **kwargs: (np.ones(16000, dtype=np.float32), 16000)

        class Encoder:
            dtype = torch.float32

            def __init__(self):
                self.kwargs = None
                self.args = None

            def __call__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                return types.SimpleNamespace(last_hidden_state=torch.ones(13, 4))

        with patch.dict(sys.modules, {"soundfile": soundfile}):
            for backend, mask_key in (("hf", "attention_mask"),
                                      ("hf", "input_features_mask"),
                                      ("qwen", "attention_mask")):
                extractor = lambda *args, **kwargs: {
                    "input_features": torch.zeros(1, 128, 100),
                    mask_key: torch.ones(1, 100, dtype=torch.long),
                }
                encoder = Encoder()
                states = encode_audio(encoder, extractor, "unused.wav", "cpu", backend)
                self.assertEqual(tuple(states.shape), (13, 4))
                if backend == "qwen":
                    self.assertEqual(tuple(encoder.args[0].shape), (128, 100))
                    self.assertEqual(encoder.kwargs["feature_lens"].tolist(), [100])
                else:
                    self.assertEqual(tuple(encoder.kwargs["input_features"].shape),
                                     (1, 128, 100))

    def test_encoder_only_sees_audio_at_current_chunk_boundary(self):
        import numpy as np
        import torch
        from model_io import encode_audio

        soundfile = types.ModuleType("soundfile")
        soundfile.read = lambda *args, **kwargs: (np.ones(48000, dtype=np.float32), 16000)
        seen = []

        def extractor(audios, **kwargs):
            seen.append(len(audios[0]))
            return {"input_features": torch.zeros(1, 128, 100),
                    "attention_mask": torch.ones(1, 100, dtype=torch.long)}

        class Encoder:
            dtype = torch.float32

            def __call__(self, **kwargs):
                return types.SimpleNamespace(last_hidden_state=torch.ones(13, 4))

        with patch.dict(sys.modules, {"soundfile": soundfile}):
            encode_audio(Encoder(), extractor, "unused.wav", "cpu", "hf",
                         end_sample=32000)
            self.assertEqual(seen, [32000])
            with self.assertRaisesRegex(ValueError, "exceeds audio length"):
                encode_audio(Encoder(), extractor, "unused.wav", "cpu", "hf",
                             end_sample=64000)

    def test_waveform_encoder_reuses_loaded_audio_for_streaming(self):
        import numpy as np
        import torch
        from model_io import encode_waveform

        seen = []

        def extractor(audios, **kwargs):
            seen.append(len(audios[0]))
            return {"input_features": torch.zeros(1, 128, 100),
                    "attention_mask": torch.ones(1, 100, dtype=torch.long)}

        class Encoder:
            dtype = torch.float32

            def __call__(self, **kwargs):
                return types.SimpleNamespace(last_hidden_state=torch.ones(13, 4))

        audio = np.ones(64000, dtype=np.float32)
        encode_waveform(Encoder(), extractor, audio[:32000], "cpu", "hf")
        encode_waveform(Encoder(), extractor, audio, "cpu", "hf")
        self.assertEqual(seen, [32000, 64000])

    def test_one_cpu_epoch_writes_usable_head_and_metrics(self):
        import torch
        import torch._dynamo  # Initialize before substituting a fake transformers module.
        import train

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

            def decode(self, ids, **kwargs):
                return "".join(chr(token) for token in ids)

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda _: Tokenizer()
        )
        fake_processor = types.SimpleNamespace(feature_extractor=object())

        def fake_encode(_, __, path, ___, backend):
            index = int(Path(path).stem)
            generator = torch.Generator().manual_seed(index)
            return torch.randn(24, 4, generator=generator, dtype=torch.float32)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest = root / "data.jsonl"
            with manifest.open("w", encoding="utf-8") as stream:
                for index in range(10):
                    (root / f"{index}.wav").write_bytes(b"fake")
                    stream.write(json.dumps({"audio": f"{index}.wav", "text": "ab"}) + "\n")
            output = root / "run"
            args = ["train.py", "--manifest", str(manifest), "--output", str(output),
                    "--epochs", "1", "--batch-size", "2", "--device", "cpu"]
            with (patch.object(sys, "argv", args),
                  patch.dict(sys.modules, {"transformers": fake_transformers}),
                  patch.object(train, "load_encoder", return_value=(object(), fake_processor, 4)),
                  patch.object(train, "encode_audio", side_effect=fake_encode)):
                train.main()
            checkpoint = torch.load(output / "best_head.pt", weights_only=True)
            metrics = json.loads((output / "best_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["vocabulary"], [None, ord("a"), ord("b")])
            self.assertEqual(metrics["samples"], 2)
            self.assertEqual(len((output / "best_predictions.jsonl").read_text().splitlines()), 2)

            # Step budget overrides epochs, and the last partial interval is evaluated.
            output2 = root / "stepped"
            args2 = args.copy()
            args2[args2.index(str(output))] = str(output2)
            args2 += ["--max-steps", "5", "--eval-every-steps", "2", "--train-eval-samples", "3"]
            with (patch.object(sys, "argv", args2),
                  patch.dict(sys.modules, {"transformers": fake_transformers}),
                  patch.object(train, "load_encoder", return_value=(object(), fake_processor, 4)),
                  patch.object(train, "encode_audio", side_effect=fake_encode)):
                train.main()
            history = [json.loads(x) for x in (output2 / "history.jsonl").read_text().splitlines()]
            self.assertEqual([x["global_step"] for x in history], [2, 4, 5])
            self.assertTrue(all(x["train_subset"]["samples"] == 3 for x in history))
            self.assertEqual(json.loads((output2 / "run.json").read_text())["planned_optimizer_steps"], 5)

            # OOV gate must fail before expensive encoder loading or cache writes.
            train_manifest = root / "train-only.jsonl"
            train_manifest.write_text(json.dumps({"audio": "0.wav", "text": "ab"}) + "\n")
            dev_manifest = root / "dev-only.jsonl"
            dev_manifest.write_text(json.dumps({"audio": "1.wav", "text": "z"}) + "\n")
            blocked = root / "blocked"
            gate_args = ["train.py", "--manifest", str(train_manifest),
                         "--dev-manifest", str(dev_manifest), "--output", str(blocked),
                         "--device", "cpu", "--max-dev-oov", "0.05"]
            with (patch.object(sys, "argv", gate_args),
                  patch.dict(sys.modules, {"transformers": fake_transformers}),
                  patch.object(train, "load_encoder") as encoder_load,
                  self.assertRaisesRegex(ValueError, "OOV")):
                train.main()
            encoder_load.assert_not_called()
            self.assertFalse(blocked.exists())

    def test_one_streaming_epoch_keeps_chunk_identity_and_audio_cutoff(self):
        import torch
        import torch._dynamo
        import train

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

            def decode(self, ids, **kwargs):
                return "".join(chr(token) for token in ids)

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda _: Tokenizer()
        )
        seen = []

        def fake_encode(_, __, path, ___, backend, end_sample=None):
            seen.append((Path(path).stem, end_sample))
            return torch.ones(24, 4) * (end_sample / 32000)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for name in ("a", "b", "c"):
                (root / f"{name}.wav").write_bytes(b"fake")
            for filename, names in (("train.jsonl", ("a", "b")),
                                    ("dev.jsonl", ("c",))):
                with (root / filename).open("w", encoding="utf-8") as stream:
                    for name in names:
                        for end, text in ((32000, "a"), (64000, "ab")):
                            stream.write(json.dumps({"audio": f"{name}.wav",
                                                     "end_sample": end,
                                                     "text": text,
                                                     "target_token_ids": [120] if end == 32000 else [120, 121]}) + "\n")
            with (root / "dev_teacher.jsonl").open("w", encoding="utf-8") as stream:
                for end, prefix, generated in ((32000, [], [120]),
                                               (64000, [], [120, 121])):
                    stream.write(json.dumps({"audio": str(root / "c.wav"),
                                             "end_sample": end,
                                             "schema_version": 2,
                                             "chunk_index": end // 32000, "chunk_samples": 32000,
                                             "previous_token_ids": [] if end == 32000 else [120],
                                             "old_draft_token_ids": [] if end == 32000 else [120],
                                             "old_prefix_match": True,
                                             "target_token_ids": prefix + generated,
                                             "prefix_token_ids": prefix,
                                             "generated_token_ids": generated}) + "\n")
            output = root / "run"
            args = ["train.py", "--manifest", str(root / "train.jsonl"),
                    "--dev-manifest", str(root / "dev.jsonl"),
                    "--dev-teacher", str(root / "dev_teacher.jsonl"),
                    "--output", str(output), "--streaming", "--epochs", "1",
                    "--selection-metric", "hybrid", "--target-token-source", "recorded",
                    "--batch-size", "2", "--device", "cpu"]
            with (patch.object(sys, "argv", args),
                  patch.dict(sys.modules, {"transformers": fake_transformers}),
                  patch.object(train, "load_encoder", return_value=(
                      object(), types.SimpleNamespace(feature_extractor=object()), 4)),
                  patch.object(train, "encode_audio", side_effect=fake_encode)):
                train.main()
            self.assertEqual(seen, [("a", 32000), ("a", 64000),
                                    ("b", 32000), ("b", 64000),
                                    ("c", 32000), ("c", 64000)])
            rows = [json.loads(line) for line in
                    (output / "best_predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["end_sample"] for row in rows], [32000, 64000])
            self.assertEqual(json.loads((output / "run.json").read_text())
                             ["config"]["streaming"], True)
            metrics = json.loads((output / "best_metrics.json").read_text())
            self.assertIn("mean_accepted_at_4", metrics)
            self.assertEqual(metrics["rollback_comparison"]["after_first"]["events"], 1)
            checkpoint = torch.load(output / "best_head.pt", weights_only=True)
            self.assertEqual(checkpoint["vocabulary"], [None, 120, 121])
            self.assertEqual(checkpoint["selection_metric"], "hybrid")


if __name__ == "__main__":
    unittest.main()
