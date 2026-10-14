"""Prepare/verify a pinned, speaker-disjoint AISHELL-1 streaming pilot (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
import wave
import zipfile
from pathlib import Path, PurePosixPath

SPLITS = ("train", "dev", "test")
RECIPE = Path(__file__).with_name("aishell_recipe.json")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def check_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError(f"source missing or size mismatch: {path}")
    if digest(path) != entry["sha256"]:
        raise ValueError(f"source checksum mismatch: {path}")


def verify_source(cache, recipe):
    for entry in recipe["files"]:
        check_file(Path(cache) / entry["name"], entry)


def download_source(cache, recipe):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for entry in recipe["files"]:
        target = cache / entry["name"]
        if target.exists():
            check_file(target, entry)
            continue
        temporary = target.with_suffix(target.suffix + ".partial")
        for attempt in range(3):
            try:
                print(f"Downloading {entry['name']} ({entry['size'] / 1e6:.1f} MB)",
                      flush=True)
                request = urllib.request.Request(entry["url"],
                                                 headers={"User-Agent": "asr-ctc-pilot/1"})
                with urllib.request.urlopen(request, timeout=120) as source, \
                        temporary.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                check_file(temporary, entry)
                temporary.replace(target)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def wav_info(data):
    with wave.open(io.BytesIO(data), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("expected 16 kHz mono PCM16 WAV")
        samples = wav.getnframes()
        if len(wav.readframes(samples)) != samples * 2:
            raise ValueError("truncated WAV")
        return samples


def choose_records(cache, recipe):
    speakers = [speaker for split in SPLITS for speaker in recipe["speakers"][split]]
    if len(speakers) != len(set(speakers)):
        raise ValueError("speaker overlap between dataset splits")
    if (recipe["per_speaker"] < 1 or recipe["min_seconds"] < 6
            or recipe["max_seconds"] < recipe["min_seconds"]):
        raise ValueError("positive sample count and 6 <= min_seconds <= max_seconds required")
    train_limit = recipe.get("train_per_speaker", recipe["per_speaker"])
    if type(train_limit) is not int or train_limit < 0:
        raise ValueError("train_per_speaker must be nonnegative; 0 means all eligible")
    frozen = {}
    if "frozen_holdout_recipe" in recipe:
        base = recipe["frozen_holdout_recipe"]
        if "frozen_holdout_recipe" in base or base["revision"] != recipe["revision"]:
            raise ValueError("frozen holdout requires a flat recipe with the same source revision")
        if any(base["speakers"][s] != recipe["speakers"][s] for s in ("dev", "test")):
            raise ValueError("frozen holdout speakers must remain unchanged")
        original = choose_records(cache, base)
        frozen = {s: original[s] for s in ("dev", "test")}
    verify_source(cache, recipe)
    transcripts = {}
    for line in (cache / "aishell_transcript_v0.8.txt").read_text("utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            uid, text = fields[0], "".join(fields[1:])
            if uid in transcripts and transcripts[uid] != text:
                raise ValueError(f"conflicting transcript: {uid}")
            transcripts[uid] = text
    selected = {}
    prior_texts = {r["text"] for rows in frozen.values() for r in rows}
    prior_hashes = {r["sha256"] for rows in frozen.values() for r in rows}
    for split in SPLITS:
        if split in frozen:
            selected[split] = frozen[split]
            continue
        rows = []
        for speaker in recipe["speakers"][split]:
            candidates = []
            with tarfile.open(cache / f"{speaker}.tar.gz", "r:gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    parts = PurePosixPath(member.name).parts
                    if (len(parts) != 3 or parts[0] not in SPLITS or parts[1] != speaker
                            or not parts[2].startswith(f"BAC009{speaker}W")):
                        raise ValueError(f"unexpected archive member: {member.name}")
                    uid = PurePosixPath(member.name).stem
                    text = transcripts.get(uid)
                    if not text or text in prior_texts:
                        continue
                    # Read members; never extract archive paths onto the filesystem.
                    data = tar.extractfile(member).read()
                    samples = wav_info(data)
                    if not recipe["min_seconds"] * 16000 <= samples <= recipe["max_seconds"] * 16000:
                        continue
                    sha = hashlib.sha256(data).hexdigest()
                    if sha in prior_hashes:
                        continue
                    candidates.append({"audio": f"audio/{split}/{speaker}/{uid}.wav",
                                       "text": text, "utterance_id": uid, "speaker_id": speaker,
                                       "source_split": parts[0], "num_samples": samples,
                                       "duration_seconds": samples / 16000, "sha256": sha,
                                       "source_member": member.name})
            candidates.sort(key=lambda row: hashlib.sha256(
                f"{recipe['seed']}:{row['utterance_id']}".encode()).hexdigest())
            limit = train_limit if split == "train" else recipe["per_speaker"]
            chosen = candidates[:limit] if limit else candidates
            if limit and len(chosen) < limit:
                raise ValueError(f"{speaker}: only {len(chosen)} eligible recordings; "
                                 f"need {limit}")
            if not chosen:
                # Only train_per_speaker=0 permits this; fixed-count splits
                # (including the frozen holdout recipe) stay strict above.
                print(f"Skipping training candidate {speaker}: 0 eligible recordings "
                      "after transcript, duration and holdout-overlap filters", flush=True)
            rows.extend(chosen)
        if not rows:
            raise ValueError(f"{split}: no eligible recordings across requested speakers")
        selected[split] = sorted(rows, key=lambda row: row["utterance_id"])
        prior_texts.update(row["text"] for row in rows)
        prior_hashes.update(row["sha256"] for row in rows)
    return selected


def summarize(selected):
    result = {}
    for split, rows in selected.items():
        samples = [row["num_samples"] for row in rows]
        ends = [n // 32000 for n in samples]
        result[split] = {
            "recordings": len(rows), "speakers": sorted({r["speaker_id"] for r in rows}),
            "audio_seconds": round(sum(samples) / 16000, 4),
            "min_seconds": min(samples) / 16000, "max_seconds": max(samples) / 16000,
            "full_2s_triggers": sum(ends),
            "third_and_later_triggers": sum(max(0, n - 2) for n in ends),
            "excluded_tail_seconds": round(sum(n % 32000 for n in samples) / 16000, 4),
            "cumulative_encoder_input_seconds": sum(n * (n + 1) for n in ends),
        }
    return {"chunk_seconds": 2.0, "splits": result}


def build_dataset(cache, output, recipe):
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    selected = choose_records(cache, recipe)
    output.mkdir(parents=True)
    for split, rows in selected.items():
        for speaker in recipe["speakers"][split]:
            wanted = {row["source_member"]: row for row in rows if row["speaker_id"] == speaker}
            with tarfile.open(cache / f"{speaker}.tar.gz", "r:gz") as tar:
                for name, row in wanted.items():
                    target = output / row["audio"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(name).read())
        with (output / f"{split}_full.jsonl").open("w", encoding="utf-8", newline="\n") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "recipe.json", recipe)
    report = summarize(selected)
    included = report["splits"]["train"]["speakers"]
    report["training_speaker_selection"] = {
        "requested": recipe["speakers"]["train"], "included": included,
        "skipped_zero_eligible": [s for s in recipe["speakers"]["train"] if s not in included],
    }
    report["source"] = "https://huggingface.co/datasets/AISHELL/AISHELL-1"
    report["revision"] = recipe["revision"]
    report["split_policy"] = "custom speaker-disjoint pilot; not official AISHELL benchmark splits"
    write_json(output / "stats.json", report)
    license_file = cache / "LICENSE-AISHELL-1.txt"
    if license_file.exists():
        shutil.copyfile(license_file, output / license_file.name)
    (output / "DATASET_README.md").write_text(
        "# AISHELL-1 CTC streaming pilot\n\n"
        "Source: https://www.openslr.org/33/ ; https://huggingface.co/datasets/AISHELL/AISHELL-1\n\n"
        "Mandarin read speech by Beijing Shell Shell Technology Co.,Ltd; Apache-2.0.\n"
        "Citation: Hui Bu et al., AISHELL-1: An Open-Source Mandarin Speech Corpus and "
        "A Speech Recognition Baseline, Oriental COCOSDA 2017.\n\n"
        "Custom speaker-disjoint train/dev/test subsets of the available mirror. "
        "These are NOT the official benchmark splits. Selected audio is unchanged; "
        "transcript word-segmentation spaces are removed. Cross-split identical "
        "normalized transcripts and identical WAV hashes are excluded.\n\n"
        "Keep this directory intact when copying offline; manifest paths are relative. "
        "Run: python prepare_aishell.py verify /path/to/this/directory\n\n"
        "*_full.jsonl contains human whole-recording transcripts, NOT chunk labels. "
        "On the target Qwen host, run collect_teacher.py separately for each split. "
        "Each trigger uses audio [0:2s], [0:4s], [0:6s], etc. "
        "The final partial chunk is excluded by the current pilot; never compare "
        "its last trigger to the human FULL-recording transcript as if time-aligned.\n\n"
        "Use dev for checkpoint choice; test only after choices are fixed. "
        "This small read-speech pilot does not establish meeting-ASR performance, "
        "long-session stability, pretraining independence, or decoding speedup.\n",
        encoding="utf-8", newline="\n")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8", newline="\n")
    return verify_dataset(output)


def verify_dataset(root):
    root = Path(root).resolve()
    expected = set()
    for line in (root / "SHA256SUMS").read_text("utf-8").splitlines():
        sha, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in expected:
            raise ValueError(f"invalid checksum path: {name}")
        expected.add(name)
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"checksum mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected | {"SHA256SUMS"}:
        raise ValueError("checksum file inventory mismatch (keep generated outputs outside dataset)")
    selected, prior_speakers, prior_texts, prior_audio = {}, set(), set(), set()
    for split in SPLITS:
        rows = [json.loads(line) for line in (root / f"{split}_full.jsonl").read_text("utf-8").splitlines()]
        if not rows:
            raise ValueError(f"empty split: {split}")
        speakers, texts, hashes = set(), set(), set()
        paths = set()
        for row in rows:
            path = (root / row["audio"]).resolve()
            if Path(row["audio"]).is_absolute() or not path.is_relative_to(root) or path in paths:
                raise ValueError("invalid or duplicate audio path")
            paths.add(path)
            samples = wav_info(path.read_bytes())
            if samples != row["num_samples"] or samples < 96000 or digest(path) != row["sha256"]:
                raise ValueError(f"audio metadata mismatch: {path}")
            speakers.add(row["speaker_id"])
            texts.add(row["text"])
            hashes.add(row["sha256"])
        if speakers & prior_speakers or texts & prior_texts or hashes & prior_audio:
            raise ValueError("cross-split speaker/text/audio leakage")
        prior_speakers.update(speakers)
        prior_texts.update(texts)
        prior_audio.update(hashes)
        selected[split] = rows
    report = summarize(selected)
    saved = json.loads((root / "stats.json").read_text("utf-8"))
    if report["splits"] != saved["splits"]:
        raise ValueError("dataset statistics mismatch")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--cache-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--recipe", type=Path, default=RECIPE)
    build.add_argument("--download", action="store_true")
    build.add_argument("--zip", type=Path, help="Optional new portable ZIP outside output directory")
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_dataset(args.directory)
    else:
        if args.zip and (args.zip.exists() or args.zip.resolve().is_relative_to(args.output_dir.resolve())):
            parser.error("ZIP must be a new file outside output directory")
        recipe = json.loads(args.recipe.read_text("utf-8"))
        if args.download:
            download_source(args.cache_dir, recipe)
        report = build_dataset(args.cache_dir, args.output_dir, recipe)
        if args.zip:
            args.zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(args.zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(args.output_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, Path(args.output_dir.name) / path.relative_to(args.output_dir))
            print(f"ZIP SHA256 {digest(args.zip)}  {args.zip}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
