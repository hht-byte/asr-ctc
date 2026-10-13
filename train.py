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
from score_teacher import read_rows, score_rows
from streaming_ctc import load_stream_manifest, split_stream_records


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B-hf",
                        help="Qwen3-ASR checkpoint or its local directory")
    parser.add_argument("--backend", choices=("hf", "qwen"), default="hf",
                        help="hf = Transformers native; qwen = original qwen-asr checkpoint")
    parser.add_argument("--manifest", required=True, help="JSONL with audio and text fields")
    parser.add_argument("--dev-manifest", help="Independent held-out JSONL (recommended)")
    parser.add_argument("--dev-teacher", help="Streaming verifier token trace for checkpoint selection")
    parser.add_argument("--target-token-source", choices=("auto", "recorded", "text"), default="auto",
                        help="auto uses recorded target IDs when all rows have them")
    parser.add_argument("--selection-metric", choices=("auto", "hybrid", "ctc"), default="auto",
                        help="auto selects incremental hybrid gain with v2 traces, CTC with legacy traces")
    parser.add_argument("--verify-budget", type=int, default=8,
                        help="Candidate token budget used for checkpoint selection")
    parser.add_argument("--streaming", action="store_true",
                        help="Train on cumulative two-second chunk events")
    parser.add_argument("--chunk-seconds", type=float, default=2.0,
                        help="Streaming trigger interval; default 2.0 seconds")
    parser.add_argument("--output", required=True, help="New, empty run directory")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8, help="Per device head batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "npu"), default="auto")
    return parser.parse_args()


def select_targets(records, tokenizer, source):
    present = ["target_token_ids" in row for row in records]
    if source == "auto":
        if any(present) and not all(present):
            raise ValueError("mixed recorded/text targets; recollect manifests or explicitly choose text")
        source = "recorded" if all(present) else "text"
    if source == "recorded":
        if not all(present):
            raise ValueError("recorded targets require target_token_ids in every train/dev event")
        tokens = [row["target_token_ids"] for row in records]
        if any(not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids)
               for ids in tokens):
            raise ValueError("invalid recorded target token IDs")
        return tokens, source
    return [tokenizer.encode(row["text"], add_special_tokens=False) for row in records], "text"


def checkpoint_score(metrics, mode, budget):
    if mode == "auto":
        mode = ("hybrid" if "rollback_comparison" in metrics else
                "ctc" if "mean_accepted_tokens" in metrics else "token_error_rate")
    if mode == "hybrid":
        if "rollback_comparison" not in metrics:
            raise ValueError("hybrid selection needs schema_version 2 teacher traces")
        after = metrics["rollback_comparison"]["after_first"]
        if not after["events"]:
            raise ValueError("hybrid selection needs second or later chunk events")
        return ((after[f"mean_delta_accepted_at_{budget}"],
                 after[f"hybrid_mean_accepted_at_{budget}"],
                 -metrics["token_error_rate"]), mode)
    if mode == "ctc":
        key = f"mean_accepted_at_{budget}"
        if key not in metrics:
            raise ValueError("CTC selection needs teacher metrics at the selected budget")
        return ((metrics[key], metrics["prefix_match_rate"], -metrics["token_error_rate"]), mode)
    return ((-metrics["token_error_rate"],), mode)


def select_dev_teacher(teachers, dev_records, auto_split):
    if not auto_split:
        return teachers
    # A full-manifest teacher file is expected with automatic recording splits.
    # Select whole recordings, never individual events that could hide gaps.
    dev_paths = {row["audio"] for row in dev_records}
    return [row for row in teachers if row.get("audio") in dev_paths]


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
        end_sample = records[index].get("end_sample")
        if end_sample is None:
            features = encode_audio(encoder, processor.feature_extractor,
                                    records[index]["audio"], device, backend)
        else:
            features = encode_audio(encoder, processor.feature_extractor,
                                    records[index]["audio"], device, backend,
                                    end_sample=end_sample)
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
                         "draft_token_ids": ids,
                         **({"end_sample": record["end_sample"]}
                            if "end_sample" in record else {})})
    return (evaluate_predictions(predictions, tokens[train_count:], texts,
                                 [r["text"] for r in records], vocabulary), rows)


def main() -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    args = arguments()
    from transformers import AutoTokenizer
    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.chunk_seconds <= 0 or args.verify_budget < 1):
        raise ValueError("epochs, batch size, learning rate and chunk seconds must be positive")
    if args.dev_teacher and not args.streaming:
        raise ValueError("--dev-teacher requires --streaming")
    if args.selection_metric != "auto" and not args.dev_teacher:
        raise ValueError("explicit --selection-metric requires --dev-teacher")
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

    chunk_samples = int(round(args.chunk_seconds * 16000))
    if args.streaming:
        load_records = lambda path, minimum: load_stream_manifest(
            path, chunk_samples, min_streams=minimum)
    else:
        load_records = lambda path, minimum: load_manifest(path, min_records=minimum)
    if args.dev_manifest:
        train_records = load_records(args.manifest, 1)
        dev_records = load_records(args.dev_manifest, 1)
        train_paths = {os.path.normcase(row["audio"]) for row in train_records}
        if train_paths & {os.path.normcase(row["audio"]) for row in dev_records}:
            raise ValueError("train and dev manifests contain overlapping audio paths")
    else:
        records = load_records(args.manifest, 2)
        splitter = split_stream_records if args.streaming else split_records
        train_records, dev_records = splitter(records, args.dev_fraction, args.seed)
    if len(train_records) < world:
        raise ValueError(f"need at least {world} training clips for {world} ranks")
    ordered_records = train_records + dev_records
    dev_teacher = read_rows(Path(args.dev_teacher)) if args.dev_teacher else None
    if dev_teacher is not None:
        dev_teacher = select_dev_teacher(dev_teacher, dev_records, auto_split=not args.dev_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_tokens, target_source = select_targets(ordered_records, tokenizer, args.target_token_source)
    selection_mode = "token_error_rate"
    if dev_teacher is not None:
        # Fail before loading the encoder, on every rank, for incompatible/incomplete traces.
        probe_rows = [{"audio": r["audio"], "end_sample": r["end_sample"],
                       "draft_token_ids": []} for r in dev_records]
        probe_metrics = score_rows(probe_rows, dev_teacher,
                                   require_rollback=args.selection_metric == "hybrid",
                                   verify_budget=args.verify_budget)
        _, selection_mode = checkpoint_score({**probe_metrics, "token_error_rate": 0},
                                               args.selection_metric, args.verify_budget)
        if selection_mode == "hybrid":
            if target_source != "recorded":
                raise ValueError("hybrid selection requires recorded target_token_ids in train/dev events")
            by_key = {(r["audio"], r["end_sample"]): r for r in dev_teacher}
            for row, ids in zip(dev_records, all_tokens[len(train_records):]):
                if ids != by_key[(row["audio"], row["end_sample"])]["target_token_ids"]:
                    raise ValueError("dev event target IDs differ from teacher trace")
    if not args.streaming and any(not row for row in all_tokens):
        raise ValueError("at least one transcript became empty after tokenization")
    vocabulary = build_vocabulary(all_tokens[:len(train_records)])
    cache_dir = output / "features"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=False)
        (output / "run.json").write_text(json.dumps({
            "model": args.model, "seed": args.seed, "train_samples": len(train_records),
            "dev_samples": len(dev_records), "world_size": world,
            "vocabulary": vocabulary,
            "target_token_source": target_source, "selection_metric": selection_mode,
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
    best_score = None

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
            if dev_teacher is not None:
                metrics.update(score_rows(rows, dev_teacher, verify_budget=args.verify_budget))
            metrics.update({"epoch": epoch, "train_loss": loss_sum / steps})
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            score, selected_mode = checkpoint_score(metrics, args.selection_metric, args.verify_budget)
            if best_score is None or score > best_score:
                best_score = score
                torch.save({"head": {key: value.detach().cpu()
                                      for key, value in head_module.state_dict().items()},
                            "hidden_size": hidden_size, "vocabulary": vocabulary,
                            "model": args.model, "backend": args.backend,
                            "streaming": args.streaming,
                            "chunk_seconds": args.chunk_seconds if args.streaming else None,
                            "selection_metric": selected_mode, "verify_budget": args.verify_budget,
                            "target_token_source": target_source,
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
