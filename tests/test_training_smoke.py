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

    def test_one_cpu_epoch_writes_usable_head_and_metrics(self):
        import torch
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


if __name__ == "__main__":
    unittest.main()
