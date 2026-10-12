"""Load either Qwen3-ASR checkpoint format and encode one clip."""

from __future__ import annotations

from pathlib import Path


def load_encoder(model_path: str, device: str, backend: str = "hf"):
    import torch

    if backend == "hf":
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        encoder = model.model.audio_tower
        hidden_size = int(model.config.audio_config.d_model)
    elif backend == "qwen":
        from qwen_asr import Qwen3ASRModel

        wrapper = Qwen3ASRModel.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=None
        )
        model = wrapper.model
        processor = wrapper.processor
        encoder = model.thinker.audio_tower
        hidden_size = int(model.config.thinker_config.audio_config.output_dim)
        del wrapper
    else:
        raise ValueError(f"unknown backend: {backend}")
    encoder.requires_grad_(False).eval()
    del model  # The decoder and projector are never used by the CTC experiment.
    return encoder.to(device), processor, hidden_size


def encode_audio(encoder, feature_extractor, path: str | Path, device: str,
                 backend: str):
    import numpy as np
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"{path}: expected 16 kHz audio, got {sample_rate} Hz")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or len(audio) < 8000 or not np.isfinite(audio).all():
        raise ValueError(f"{path}: expected finite mono audio of at least 0.5 s")
    inputs = feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt",
        return_attention_mask=True, padding="longest", truncation=False
    )
    features = inputs["input_features"].to(device=device, dtype=encoder.dtype)
    mask_key = "input_features_mask" if "input_features_mask" in inputs else "attention_mask"
    mask = inputs[mask_key].to(device=device)
    with torch.inference_mode():
        if backend == "hf":
            # Transformers-native encoder takes padded [batch, mel, time].
            encoded = encoder(input_features=features, input_features_mask=mask)
        elif backend == "qwen":
            # Original qwen-asr encoder takes one unpadded [mel, time] clip.
            valid_frames = int(mask[0].sum().item())
            encoded = encoder(features[0, :, :valid_frames],
                              feature_lens=torch.tensor([valid_frames], device=device))
        else:
            raise ValueError(f"unknown backend: {backend}")
        states = encoded.last_hidden_state
    if states.ndim != 2 or states.shape[0] == 0:
        raise RuntimeError(f"{path}: unexpected audio encoder output shape {tuple(states.shape)}")
    return states.detach().to(device="cpu", dtype=torch.float16)
