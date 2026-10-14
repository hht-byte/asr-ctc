"""Strict partial checkpoint loading must never leave random encoder weights."""
import importlib.util
import json
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(n) for n in ("torch", "safetensors"))


@unittest.skipUnless(HAS_DEPS, "requires torch and safetensors")
class AudioCheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        self.torch = torch
        self.save = save_file
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.encoder = torch.nn.Linear(3, 2)
        self.weights = {"thinker.audio_tower.weight": torch.arange(6.).reshape(2, 3),
                        "thinker.audio_tower.bias": torch.tensor([7., 8.]),
                        "thinker.model.decoder.weight": torch.ones(9)}

    def test_single_file_copies_exact_values_without_reading_decoder(self):
        from qwen_audio import load_audio_weights
        from safetensors import safe_open
        self.save(self.weights, self.root / "model.safetensors")
        reads = []

        class Reader:
            def __init__(inner, *args, **kwargs):
                inner.reader = safe_open(*args, **kwargs)
            def __enter__(inner):
                inner.reader.__enter__()
                return inner
            def __exit__(inner, *args):
                return inner.reader.__exit__(*args)
            def keys(inner):
                return inner.reader.keys()
            def get_tensor(inner, key):
                reads.append(key)
                return inner.reader.get_tensor(key)

        with patch("safetensors.safe_open", Reader):
            count = load_audio_weights(self.encoder, self.root)
        self.assertEqual(count, 2)
        self.assertEqual(set(reads), {k for k in self.weights if "audio_tower" in k})
        self.torch.testing.assert_close(self.encoder.weight, self.weights["thinker.audio_tower.weight"])
        self.torch.testing.assert_close(self.encoder.bias, self.weights["thinker.audio_tower.bias"])

    def test_shards_skip_absent_decoder_shard_and_copy_bfloat16(self):
        from qwen_audio import load_audio_weights
        mapping = {k: f"part-{i}.safetensors" for i, k in enumerate(self.weights)}
        for key, filename in list(mapping.items())[:2]:
            self.save({key: self.weights[key]}, self.root / filename)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        self.encoder.to(dtype=self.torch.bfloat16)
        self.assertEqual(load_audio_weights(self.encoder, self.root), 2)
        self.torch.testing.assert_close(self.encoder.weight.float(), self.weights["thinker.audio_tower.weight"])

    def test_missing_unexpected_and_wrong_shape_fail(self):
        from qwen_audio import load_audio_weights
        variants = [
            ({"thinker.audio_tower.weight": self.weights["thinker.audio_tower.weight"]}, "missing"),
            ({**self.weights, "thinker.audio_tower.extra": self.torch.zeros(1)}, "unexpected"),
            ({**self.weights, "thinker.audio_tower.weight": self.torch.zeros(5, 2)}, "shape"),
        ]
        for weights, message in variants:
            with self.subTest(message=message):
                self.save(weights, self.root / "model.safetensors")
                with self.assertRaisesRegex(ValueError, message):
                    load_audio_weights(self.encoder, self.root)

    def test_index_cannot_hide_missing_tensor_or_escape_directory(self):
        from qwen_audio import load_audio_weights
        index = self.root / "model.safetensors.index.json"
        mapping = {k: "part.safetensors" for k in self.weights if "audio_tower" in k}
        index.write_text(json.dumps({"weight_map": mapping}))
        self.save({"unrelated": self.torch.zeros(1)}, self.root / "part.safetensors")
        with self.assertRaisesRegex(ValueError, "index|missing"):
            load_audio_weights(self.encoder, self.root)
        index.write_text(json.dumps({"weight_map": {k: "../outside.safetensors" for k in mapping}}))
        with self.assertRaisesRegex(ValueError, "path|outside"):
            load_audio_weights(self.encoder, self.root)

    def test_public_loader_uses_only_audio_config_and_freezes_encoder(self):
        from model_io import load_encoder
        torch = self.torch
        self.save(self.weights, self.root / "model.safetensors")
        (self.root / "config.json").write_text(json.dumps({
            "model_type": "qwen3_asr", "thinker_config": {
                "audio_config": {"output_dim": 2}, "text_config": {"must_not_construct": True}}}))

        class Audio(torch.nn.Linear):
            def __init__(inner, config):
                super().__init__(3, 2)
                config._attn_implementation = "sdpa"

        def component(path, **kwargs):
            self.assertEqual(Path(path), self.root)
            self.assertTrue(kwargs["local_files_only"])
            return object()

        modules = {n: types.ModuleType(n) for n in (
            "qwen_asr", "qwen_asr.core", "qwen_asr.core.transformers_backend",
            "qwen_asr.core.transformers_backend.configuration_qwen3_asr",
            "qwen_asr.core.transformers_backend.modeling_qwen3_asr", "transformers")}
        modules["qwen_asr.core.transformers_backend.configuration_qwen3_asr"].Qwen3ASRAudioEncoderConfig = types.SimpleNamespace
        modules["qwen_asr.core.transformers_backend.modeling_qwen3_asr"].Qwen3ASRAudioEncoder = Audio
        modules["transformers"].WhisperFeatureExtractor = types.SimpleNamespace(from_pretrained=component)
        modules["transformers"].Qwen2TokenizerFast = types.SimpleNamespace(from_pretrained=component)
        # No full-model/config factories are exposed: using one must fail this test.
        with patch.dict(sys.modules, modules):
            encoder, processor, hidden = load_encoder(str(self.root), "cpu", "qwen")
        self.assertEqual(hidden, 2)
        self.assertFalse(encoder.training)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        self.assertEqual(encoder.weight.dtype, torch.bfloat16)
        self.assertEqual(encoder.ctc_load_info["loaded_audio_tensors"], 2)
        self.assertIsNotNone(processor.tokenizer)
        torch.testing.assert_close(encoder.weight.float(), self.weights["thinker.audio_tower.weight"])


if __name__ == "__main__":
    unittest.main()
