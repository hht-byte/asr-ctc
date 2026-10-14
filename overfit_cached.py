"""Single-device CTC memorization diagnostic using an existing run's train cache.

No encoder, tokenizer, audio, dev or test inference is needed. This is a training
chain diagnostic, not a generalization or speculative speedup measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

from ctc_pilot import collapse_ctc, edit_distance, longest_common_prefix, minimum_ctc_frames


def load_subset(run_dir, recordings=8, seed=42):
    import torch

    root = Path(run_dir)
    raw = json.loads((root / "run.json").read_text("utf-8"))
    if raw.get("target_token_source") != "recorded":
        raise ValueError("diagnostic requires an existing run with recorded target IDs")
    vocabulary = raw["vocabulary"]
    if (len(vocabulary) < 2 or vocabulary[0] is not None
            or any(type(t) is not int or t < 0 for t in vocabulary[1:])
            or len(set(vocabulary[1:])) != len(vocabulary) - 1):
        raise ValueError("invalid original CTC vocabulary")
    local = {t: i for i, t in enumerate(vocabulary) if t is not None}
    rows = raw["split"]["train"]
    names = sorted({r["audio"] for r in rows})
    if type(recordings) is not int or not 1 <= recordings <= len(names):
        raise ValueError(f"recordings must be in 1..{len(names)}")
    random.Random(seed).shuffle(names)
    chosen = set(names[:recordings])
    items = []
    hidden = None
    for index, row in enumerate(rows):
        if row["audio"] not in chosen:
            continue
        ids = row.get("target_token_ids")
        if not isinstance(ids, list) or any(type(t) is not int or t not in local for t in ids):
            raise ValueError(f"train event {index}: missing IDs or targets outside original vocabulary")
        features = torch.load(root / "features" / f"{index:06d}.pt",
                              map_location="cpu", weights_only=True)["features"].float()
        if features.ndim != 2 or not all(features.shape) or not torch.isfinite(features).all():
            raise ValueError(f"train event {index}: expected nonempty finite [frames, hidden] features")
        if hidden is not None and features.shape[1] != hidden:
            raise ValueError("cached feature hidden dimensions differ")
        hidden = features.shape[1]
        if len(features) < minimum_ctc_frames(ids):
            raise ValueError(f"train event {index}: not enough CTC frames for targets")
        items.append({"index": index, "audio": row["audio"],
                      "end_sample": row.get("end_sample"), "tokens": ids,
                      "labels": torch.tensor([local[t] for t in ids], dtype=torch.long),
                      "features": features})
    if not items or not any(x["tokens"] for x in items):
        raise ValueError("selected recordings have no nonempty training targets")
    metadata = {"source_run": str(root.resolve()), "recordings": recordings,
                "samples": len(items), "hidden_size": hidden, "vocabulary": vocabulary,
                "feature_rms": math.sqrt(sum(float(x["features"].square().sum()) for x in items)
                                         / sum(x["features"].numel() for x in items)),
                "min_frames": min(len(x["features"]) for x in items),
                "max_frames": max(len(x["features"]) for x in items),
                "selected_train_indices": [x["index"] for x in items]}
    return metadata, items


def measure(head, items, vocabulary, device):
    import torch

    head.eval()
    errors = targets = exact = prefix = blank_frames = frames = 0
    predictions = []
    with torch.inference_mode():
        for item in items:
            logits = head(item["features"].to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite CTC logits")
            local_ids = logits.argmax(-1).cpu().tolist()
            ids = collapse_ctc(local_ids, vocabulary)
            truth = item["tokens"]
            errors += edit_distance(ids, truth)
            targets += len(truth)
            exact += ids == truth
            prefix += longest_common_prefix(ids, truth)
            blank_frames += local_ids.count(0)
            frames += len(local_ids)
            predictions.append({"train_index": item["index"], "draft_token_ids": ids,
                                "target_token_ids": truth})
    return {"samples": len(items), "token_error_rate": errors / max(1, targets),
            "exact_sequence_rate": exact / len(items),
            "mean_prefix_tokens": prefix / len(items),
            "mean_draft_tokens": sum(len(x["draft_token_ids"]) for x in predictions) / len(items),
            "blank_frame_rate": blank_frames / frames}, predictions


def run_probe(run_dir, output, device, *, recordings=8, steps=1000,
              batch_size=8, learning_rate=3e-4, eval_every=50, seed=42):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("use python on one device, not a multi-process torchrun")
    if (steps < 1 or batch_size < 1 or eval_every < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("steps, batch size, eval interval and learning rate must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("diagnostic output must be new or empty")
    metadata, items = load_subset(run_dir, recordings, seed)
    torch.manual_seed(seed)
    # Initialize on CPU first so CPU/NPU runs start from the same parameters.
    head = torch.nn.Linear(metadata["hidden_size"], len(metadata["vocabulary"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    output.mkdir(parents=True, exist_ok=True)
    metadata.update({"device": device, "seed": seed, "steps_budget": steps,
                     "batch_size": batch_size, "learning_rate": learning_rate,
                     "initialization": "fresh_linear_head_original_vocabulary",
                     "diagnostic_only": True})
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def report(step, diagnostics):
        metrics, predictions = measure(head, items, metadata["vocabulary"], device)
        passed = metrics["token_error_rate"] <= 0.01 and metrics["exact_sequence_rate"] >= 0.95
        metrics.update({"step": step, **diagnostics,
                        "status": "memorized_subset" if passed else
                                  "budget_exhausted" if step == steps else "running"})
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        (output / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in predictions), encoding="utf-8")
        return metrics

    result = report(0, {})
    rng = random.Random(seed)
    order = []
    for step in range(1, steps + 1):
        if result["status"] == "memorized_subset":
            break
        if not order:
            order = list(range(len(items)))
            rng.shuffle(order)
        selected, order = order[:batch_size], order[batch_size:]
        batch = [items[i] for i in selected]
        features = pad_sequence([x["features"] for x in batch], batch_first=True).to(device)
        lengths = torch.tensor([len(x["features"]) for x in batch], dtype=torch.long)
        labels = torch.cat([x["labels"] for x in batch])
        label_lengths = torch.tensor([len(x["labels"]) for x in batch], dtype=torch.long)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        log_probs = head(features).log_softmax(-1).transpose(0, 1).cpu()
        loss = criterion(log_probs, labels, lengths, label_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite CTC loss")
        loss.backward()
        if any(p.grad is None for p in head.parameters()):
            raise RuntimeError("CTC backward did not produce gradients for the head")
        gradient_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in head.parameters()))
        if not math.isfinite(gradient_norm):
            raise RuntimeError("non-finite CTC gradient")
        log_step = step == 1 or step % eval_every == 0 or step == steps
        before = head.weight.detach().clone() if log_step else None
        optimizer.step()
        if log_step:
            update = float((head.weight.detach() - before).abs().max())
            if not math.isfinite(update) or (step == 1 and (gradient_norm == 0 or update == 0)):
                raise RuntimeError("first update has zero gradient/update or update is non-finite")
            result = report(step, {"batch_loss": float(loss.detach()),
                                   "gradient_norm": gradient_norm, "weight_update_max_abs": update})
    # Deliberately not named best_head.pt: this is not a deployment checkpoint.
    torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "vocabulary": metadata["vocabulary"], "hidden_size": metadata["hidden_size"],
                "step": result["step"], "diagnostic_only": True}, output / "last_head.pt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Original run with run.json and features/")
    parser.add_argument("--output", required=True, help="New diagnostic directory")
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--recordings", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        parser.error("run with python on a single device")
    from train import select_device
    device = select_device(args.device, 0)
    run_probe(args.run, args.output, device, recordings=args.recordings, steps=args.steps,
              batch_size=args.batch_size, learning_rate=args.learning_rate,
              eval_every=args.eval_every, seed=args.seed)


if __name__ == "__main__":
    main()
