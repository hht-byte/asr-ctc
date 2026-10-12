"""Frozen Qwen3-ASR audio encoder + trainable CTC head pilot.

Run with python for one device or torchrun for eight Ascend 910B4 devices.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

from ctc_pilot import (build_vocabulary, collapse_ctc, evaluate_predictions,
                       load_manifest, minimum_ctc_frames, split_records)
from model_io import encode_audio, load_encoder


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B-hf",
                        help="Qwen3-ASR checkpoint or its local directory")
    parser.add_argument("--backend", choices=("hf", "qwen"), default="hf",
                        help="hf = Transformers native; qwen = original qwen-asr checkpoint")
    parser.add_argument("--manifest", required=True, help="JSONL with audio and text fields")
    parser.add_argument("--dev-manifest", help="Independent held-out JSONL (recommended)")
    parser.add_argument("--output", required=True, help="New, empty run directory")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8, help="Per device head batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    return parser.parse_args()


def select_device(requested: str, local_rank: int) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    try:
        import torch_npu  # noqa: F401 - registers torch.npu
        available = torch.npu.is_available()
    except ImportError:
        available = False
    if requested == "npu" and not available:
        raise RuntimeError("--device npu requires a working torch_npu installation")
    if available:
        torch.npu.set_device(local_rank)
        return f"npu:{local_rank}"
    return "cpu"


def precompute(records, cache_dir, encoder, processor, device, backend, rank, world):
    import torch

    for index in range(rank, len(records), world):
        features = encode_audio(encoder, processor.feature_extractor,
                                records[index]["audio"], device, backend)
        torch.save({"features": features}, cache_dir / f"{index:06d}.pt")
        if rank == 0 and index % max(world * 20, 1) == 0:
            print(f"encoded {index + 1}/{len(records)} clips", flush=True)


def make_loader(indices, all_tokens, cache_dir, vocabulary, batch_size,
                rank, world, shuffle, seed):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset, DistributedSampler

    token_to_local = {token: i for i, token in enumerate(vocabulary) if token is not None}

    class CachedDataset(Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, position):
            index = indices[position]
            features = torch.load(cache_dir / f"{index:06d}.pt", weights_only=True)["features"]
            labels = torch.tensor([token_to_local[t] for t in all_tokens[index]],
                                  dtype=torch.long)
            return features, labels

    def collate(items):
        features, labels = zip(*items)
        lengths = torch.tensor([len(row) for row in features], dtype=torch.long)
        label_lengths = torch.tensor([len(row) for row in labels], dtype=torch.long)
        return (pad_sequence(features, batch_first=True), lengths,
                torch.cat(labels), label_lengths)

    dataset = CachedDataset()
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=shuffle, seed=seed,
                                 drop_last=shuffle) if world > 1 else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=shuffle and sampler is None, collate_fn=collate,
                        num_workers=0)
    return loader, sampler


def evaluate(head, cache_dir, train_count, records, tokens, tokenizer, vocabulary, device):
    import torch

    head.eval()
    predictions, texts, rows = [], [], []
    with torch.inference_mode():
        for offset, record in enumerate(records):
            index = train_count + offset
            features = torch.load(cache_dir / f"{index:06d}.pt", weights_only=True)["features"]
            logits = head(features.to(device=device, dtype=torch.float32))
            local_ids = logits.argmax(-1).cpu().tolist()
            ids = collapse_ctc(local_ids, vocabulary)
            text = tokenizer.decode(ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            predictions.append(ids)
            texts.append(text)
            rows.append({"audio": record["audio"], "reference": record["text"],
                         "draft": text, "reference_token_ids": tokens[index],
                         "draft_token_ids": ids})
    return (evaluate_predictions(predictions, tokens[train_count:], texts,
                                 [r["text"] for r in records], vocabulary), rows)


def main() -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoTokenizer

    args = arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    output = Path(args.output).resolve()
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = select_device(args.device, local_rank)
    if world > 1:
        dist.init_process_group("hccl" if device.startswith("npu") else "gloo")
    occupied = output.exists() and any(output.iterdir()) if rank == 0 else False
    if world > 1:
        occupied_flag = torch.tensor([int(occupied)], dtype=torch.int32, device=device)
        dist.broadcast(occupied_flag, src=0)
        occupied = bool(occupied_flag.item())
    if occupied:
        raise FileExistsError(f"Run directory must be new or empty: {output}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.dev_manifest:
        train_records = load_manifest(args.manifest, min_records=1)
        dev_records = load_manifest(args.dev_manifest, min_records=1)
        train_paths = {os.path.normcase(row["audio"]) for row in train_records}
        if train_paths & {os.path.normcase(row["audio"]) for row in dev_records}:
            raise ValueError("train and dev manifests contain overlapping audio paths")
    else:
        records = load_manifest(args.manifest)
        train_records, dev_records = split_records(records, args.dev_fraction, args.seed)
    if len(train_records) < world:
        raise ValueError(f"need at least {world} training clips for {world} ranks")
    ordered_records = train_records + dev_records
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_tokens = [tokenizer.encode(row["text"], add_special_tokens=False)
                  for row in ordered_records]
    if any(not row for row in all_tokens):
        raise ValueError("at least one transcript became empty after tokenization")
    vocabulary = build_vocabulary(all_tokens[:len(train_records)])
    cache_dir = output / "features"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=False)
        (output / "run.json").write_text(json.dumps({
            "model": args.model, "seed": args.seed, "train_samples": len(train_records),
            "dev_samples": len(dev_records), "world_size": world,
            "vocabulary": vocabulary,
            "split": {"train": train_records, "dev": dev_records},
            "config": vars(args),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if world > 1:
        dist.barrier()

    encoder, processor, hidden_size = load_encoder(args.model, device, args.backend)
    precompute(ordered_records, cache_dir, encoder, processor, device,
               args.backend, rank, world)
    if world > 1:
        dist.barrier()
    del encoder, processor
    gc.collect()
    if device.startswith("npu"):
        torch.npu.empty_cache()

    # CTC requires extra blank frames for consecutive identical labels.
    for index in range(len(train_records)):
        frames = torch.load(cache_dir / f"{index:06d}.pt", weights_only=True)["features"].shape[0]
        if frames < minimum_ctc_frames(all_tokens[index]):
            raise ValueError(f"{ordered_records[index]['audio']}: {frames} frames cannot fit "
                             f"{minimum_ctc_frames(all_tokens[index])} CTC frames")

    head = torch.nn.Linear(hidden_size, len(vocabulary), dtype=torch.float32).to(device)
    if world > 1:
        head = DistributedDataParallel(
            head, device_ids=[local_rank] if device.startswith("npu") else None
        )
    head_module = head.module if world > 1 else head
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    train_indices = list(range(len(train_records)))
    loader, sampler = make_loader(train_indices, all_tokens, cache_dir, vocabulary,
                                  args.batch_size, rank, world, True, args.seed)
    best_error = float("inf")

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        head.train()
        loss_sum = 0.0
        steps = 0
        for features, lengths, labels, label_lengths in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.to(device=device, dtype=torch.float32))
            # CPU CTCLoss works across torch_npu versions; autograd copies the
            # gradient back to the NPU head. The frozen encoder has no graph.
            log_probs = logits.float().log_softmax(-1).transpose(0, 1).cpu()
            loss = criterion(log_probs, labels, lengths, label_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite CTC loss; check labels and frame lengths")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        if world > 1:
            totals = torch.tensor([loss_sum, steps], device=device, dtype=torch.float32)
            dist.all_reduce(totals)
            loss_sum, steps = totals.cpu().tolist()
        if rank == 0:
            metrics, rows = evaluate(head_module, cache_dir, len(train_records),
                                     dev_records, all_tokens, tokenizer, vocabulary, device)
            metrics.update({"epoch": epoch, "train_loss": loss_sum / steps})
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if metrics["token_error_rate"] < best_error:
                best_error = metrics["token_error_rate"]
                torch.save({"head": {key: value.detach().cpu()
                                      for key, value in head_module.state_dict().items()},
                            "hidden_size": hidden_size, "vocabulary": vocabulary,
                            "model": args.model, "backend": args.backend,
                            "epoch": epoch}, output / "best_head.pt")
                (output / "best_metrics.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                with (output / "best_predictions.jsonl").open("w", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
