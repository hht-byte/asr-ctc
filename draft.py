"""Emit Qwen tokenizer IDs and decoded text from a trained CTC draft head."""

from __future__ import annotations

import argparse
import json

from ctc_pilot import collapse_ctc
from model_io import encode_audio, load_encoder
from train import select_device


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="best_head.pt from train.py")
    parser.add_argument("--audio", required=True, help="16 kHz audio file")
    parser.add_argument("--model", help="Override checkpoint model path, for offline relocation")
    parser.add_argument("--backend", choices=("hf", "qwen"),
                        help="Override checkpoint backend")
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_path = args.model or checkpoint["model"]
    backend = args.backend or checkpoint.get("backend", "hf")
    device = select_device(args.device, 0)
    encoder, processor, hidden_size = load_encoder(model_path, device, backend)
    if hidden_size != checkpoint["hidden_size"]:
        raise ValueError("Encoder hidden size differs from the CTC checkpoint")
    states = encode_audio(encoder, processor.feature_extractor, args.audio,
                          device, backend)
    head = torch.nn.Linear(hidden_size, len(checkpoint["vocabulary"]),
                           dtype=torch.float32).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    with torch.inference_mode():
        local_ids = head(states.to(device=device, dtype=torch.float32)).argmax(-1).cpu().tolist()
    token_ids = collapse_ctc(local_ids, checkpoint["vocabulary"])
    print(json.dumps({"audio": args.audio, "model": model_path, "backend": backend,
                      "token_ids": token_ids,
                      "text": processor.tokenizer.decode(
                          token_ids, skip_special_tokens=False,
                          clean_up_tokenization_spaces=False
                      )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
