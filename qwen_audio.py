"""Load the original Qwen audio tower without constructing its text decoder."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

AUDIO_PREFIX = "thinker.audio_tower."


def _shard_path(root, name):
    # Official shard names are filenames, never arbitrary paths from an index.
    if not isinstance(name, str) or "/" in name or "\\" in name or ":" in name or not name.endswith(".safetensors"):
        raise ValueError(f"unsupported checkpoint shard path: {name!r}")
    return root / name


def load_audio_weights(encoder, model_dir):
    """Copy only audio tensors, requiring an exact match to encoder.state_dict()."""
    import torch
    from safetensors import safe_open

    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    targets = encoder.state_dict()
    expected = {AUDIO_PREFIX + key for key in targets}
    if index_path.is_file():
        index = json.loads(index_path.read_text("utf-8"))["weight_map"]
        mapping = {k: v for k, v in index.items() if k.startswith(AUDIO_PREFIX)}
    else:
        filename = "model.safetensors"
        if not (root / filename).is_file():
            raise FileNotFoundError(f"{root}: expected model.safetensors or model.safetensors.index.json")
        with safe_open(str(root / filename), framework="pt", device="cpu") as reader:
            mapping = {k: filename for k in reader.keys() if k.startswith(AUDIO_PREFIX)}
    missing, unexpected = expected - mapping.keys(), mapping.keys() - expected
    if missing or unexpected:
        raise ValueError(f"audio checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    shards = {}
    for key, filename in mapping.items():
        path = _shard_path(root, filename)
        shards.setdefault(path, set()).add(key)
    loaded = 0
    for path, keys in sorted(shards.items()):
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            actual = {k for k in reader.keys() if k.startswith(AUDIO_PREFIX)}
            if actual != keys:
                raise ValueError(f"audio shard/index mismatch in {path}: missing={sorted(keys-actual)}, unexpected={sorted(actual-keys)}")
            for key in sorted(keys):
                value = reader.get_tensor(key)
                target = targets[key[len(AUDIO_PREFIX):]]
                if value.shape != target.shape:
                    raise ValueError(f"audio tensor shape mismatch: {key}: {tuple(value.shape)} != {tuple(target.shape)}")
                if value.is_floating_point() != target.is_floating_point():
                    raise ValueError(f"audio tensor dtype mismatch: {key}: {value.dtype} vs {target.dtype}")
                with torch.no_grad():
                    target.copy_(value)
                loaded += 1
    return loaded


def _model_directory(model_path):
    root = Path(model_path)
    if root.is_dir():
        return root
    if root.is_absolute() or str(model_path).startswith("."):
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    from huggingface_hub import snapshot_download
    # Keep Hub model IDs working; for offline/NPU runs use a local checkpoint.
    return Path(snapshot_download(str(model_path), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))


def load_qwen_audio(model_path):
    import torch
    from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    root = _model_directory(model_path)
    raw = json.loads((root / "config.json").read_text("utf-8"))
    if raw.get("model_type") != "qwen3_asr":
        raise ValueError("--backend qwen requires the original qwen3_asr checkpoint")
    audio = raw.get("thinker_config", {}).get("audio_config")
    if not isinstance(audio, dict) or not audio:
        raise ValueError("checkpoint is missing thinker_config.audio_config")
    # No Qwen3ASRConfig, thinker or text model is instantiated here.
    config = Qwen3ASRAudioEncoderConfig(**audio)
    encoder = Qwen3ASRAudioEncoder(config).to(dtype=torch.bfloat16)
    count = load_audio_weights(encoder, root)
    # CTC needs only these components, not the ASR processor's chat/template API.
    processor = SimpleNamespace(
        feature_extractor=WhisperFeatureExtractor.from_pretrained(root, local_files_only=True),
        tokenizer=Qwen2TokenizerFast.from_pretrained(root, local_files_only=True),
    )
    encoder.ctc_load_info = {"encoder_only": True, "loaded_audio_tensors": count,
                             "text_decoder_loaded": False,
                             "attention_implementation": config._attn_implementation}
    return encoder, processor, int(config.output_dim)
