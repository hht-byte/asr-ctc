"""One-clip checkpoint and audio-encoder contract check before training."""

from __future__ import annotations

import argparse
import json

from model_io import encode_audio, load_encoder
from train import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("hf", "qwen"), required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device, 0)
    encoder, processor, hidden_size = load_encoder(args.model, device, args.backend)
    states = encode_audio(encoder, processor.feature_extractor, args.audio,
                          device, args.backend)
    if states.shape[1] != hidden_size or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("Encoder dimension or frozen-parameter check failed")
    print(json.dumps({"backend": args.backend, "device": device,
                      "frames": states.shape[0], "hidden_size": hidden_size,
                      "encoder_frozen": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
