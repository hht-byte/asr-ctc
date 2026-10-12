"""Replay cumulative two-second audio prefixes through a streaming CTC head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ctc_pilot import collapse_ctc
from model_io import encode_waveform, load_encoder
from streaming_ctc import load_stream_manifest
from train import select_device


def full_chunk_ends(audio_samples: int, chunk_samples: int) -> list[int]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    return list(range(chunk_samples, audio_samples + 1, chunk_samples))


def generate_drafts(audio, ends, head, encoder, feature_extractor, device,
                    backend, vocabulary, audio_path: str) -> list[dict]:
    import torch

    rows = []
    chunk_seconds = ends[0] / 16000 if ends else None
    previous = 0
    for end in ends:
        if type(end) is not int or end <= previous or end > len(audio):
            raise ValueError(f"invalid cumulative chunk boundary: {end}")
        previous = end
        if device.startswith("npu"):
            torch.npu.synchronize()
        start = time.perf_counter()
        states = encode_waveform(encoder, feature_extractor, audio[:end],
                                 device, backend)
        head_start = time.perf_counter()
        with torch.inference_mode():
            local_ids = head(states.to(device=device, dtype=torch.float32)).argmax(-1)
            local_ids = local_ids.cpu().tolist()
        if device.startswith("npu"):
            torch.npu.synchronize()
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start) * 1000
        rows.append({"audio": audio_path, "end_sample": end,
                     "draft_token_ids": collapse_ctc(local_ids, vocabulary),
                     "chunk_seconds": chunk_seconds,
                     "head_ms": (end_time - head_start) * 1000,
                     "encoder_head_ms": elapsed_ms})
    return rows


def read_audio(path: str):
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"{path}: expected 16 kHz audio, got {sample_rate} Hz")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"{path}: expected finite mono audio")
    return audio


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Streaming best_head.pt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="One complete 16 kHz recording")
    source.add_argument("--manifest", help="Held-out streaming event JSONL")
    parser.add_argument("--output", help="JSONL output; defaults to stdout")
    parser.add_argument("--model", help="Override checkpoint model path")
    parser.add_argument("--backend", choices=("hf", "qwen"))
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not checkpoint.get("streaming"):
        raise ValueError("stream_draft requires a checkpoint trained with --streaming")
    chunk_samples = int(round(checkpoint["chunk_seconds"] * 16000))
    if args.audio:
        path = str(Path(args.audio).resolve())
        audio_by_path = {path: None}
        event_ends = None
    else:
        events = load_stream_manifest(args.manifest, chunk_samples, min_streams=1)
        audio_by_path = dict.fromkeys(row["audio"] for row in events)
        event_ends = {path: [row["end_sample"] for row in events
                             if row["audio"] == path] for path in audio_by_path}
    device = select_device(args.device, 0)
    backend = args.backend or checkpoint["backend"]
    encoder, processor, hidden_size = load_encoder(args.model or checkpoint["model"],
                                                    device, backend)
    if hidden_size != checkpoint["hidden_size"]:
        raise ValueError("Encoder hidden size differs from the CTC checkpoint")
    head = torch.nn.Linear(hidden_size, len(checkpoint["vocabulary"]),
                           dtype=torch.float32).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8") if output else None
    try:
        for path in audio_by_path:
            audio = read_audio(path)
            ends = event_ends[path] if event_ends is not None else full_chunk_ends(
                len(audio), chunk_samples)
            for row in generate_drafts(audio, ends, head, encoder,
                                       processor.feature_extractor, device, backend,
                                       checkpoint["vocabulary"], path):
                line = json.dumps(row, ensure_ascii=False)
                if stream:
                    stream.write(line + "\n")
                else:
                    print(line)
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
