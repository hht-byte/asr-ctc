"""Portable dataset preparation and leakage/corruption checks (no model needed)."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

from prepare_aishell import build_dataset, verify_dataset, verify_source, choose_records


class DatasetTests(unittest.TestCase):
    def fixture(self, root, include_short_candidate=False):
        cache = root / "cache"
        cache.mkdir()
        recipe = {"revision": "fixture", "seed": 42, "per_speaker": 1,
                  "min_seconds": 6, "max_seconds": 30,
                  "speakers": {"train": ["S0001"], "dev": ["S0002"],
                               "test": ["S0003"]}, "files": []}
        transcript = []
        for index in range(1, 5 if include_short_candidate else 4):
            speaker = f"S{index:04d}"
            archive = cache / f"{speaker}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for j, duration in enumerate((2,) if index == 4 else (2, 6.1, 8.1)):
                    uid = f"BAC009{speaker}W{j:04d}"
                    transcript.append(f"{uid} 测 试 {index} {j}")
                    data = io.BytesIO()
                    with wave.open(data, "wb") as wav:
                        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        wav.writeframes(bytes([index, j]) * int(duration * 16000))
                    member = tarfile.TarInfo(f"train/{speaker}/{uid}.wav")
                    member.size = len(data.getvalue())
                    tar.addfile(member, io.BytesIO(data.getvalue()))
            recipe["files"].append(self.file_info(archive))
        path = cache / "aishell_transcript_v0.8.txt"
        path.write_text("\n".join(transcript), encoding="utf-8")
        recipe["files"].append(self.file_info(path))
        return cache, recipe

    @staticmethod
    def file_info(path):
        return {"name": path.name, "path": path.name, "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_deterministic_portable_speaker_disjoint_and_full_chunks(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root)
            one, two = root / "one", root / "two"
            build_dataset(cache, one, recipe)
            build_dataset(cache, two, recipe)
            self.assertEqual((one / "SHA256SUMS").read_bytes(),
                             (two / "SHA256SUMS").read_bytes())
            for split in ("train", "dev", "test"):
                row = json.loads((one / f"{split}_full.jsonl").read_text("utf-8"))
                self.assertFalse(Path(row["audio"]).is_absolute())
                self.assertGreaterEqual(row["num_samples"], 96000)
                self.assertNotIn(" ", row["text"])
            moved = root / "moved"
            one.rename(moved)
            report = verify_dataset(moved)
            self.assertEqual(report["splits"]["train"]["recordings"], 1)
            self.assertGreaterEqual(report["splits"]["train"]["full_2s_triggers"], 3)
            audio = next(moved.rglob("*.wav"))
            audio.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_dataset(moved)

    def test_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            cache, recipe = self.fixture(Path(directory))
            (cache / "S0001.tar.gz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum|size"):
                verify_source(cache, recipe)

    def test_expansion_freezes_original_holdouts_and_adds_training_audio(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root)
            build_dataset(cache, root / "base", recipe)
            expanded = {**recipe, "train_per_speaker": 0,
                        "frozen_holdout_recipe": recipe}
            report = build_dataset(cache, root / "expanded", expanded)
            self.assertEqual(report["splits"]["train"]["recordings"], 2)
            for split in ("dev", "test"):
                self.assertEqual((root / "base" / f"{split}_full.jsonl").read_bytes(),
                                 (root / "expanded" / f"{split}_full.jsonl").read_bytes())

    def test_expanded_training_excludes_frozen_holdout_transcripts(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root)
            selected = choose_records(cache, recipe)
            chosen_id = selected["train"][0]["utterance_id"]
            candidate_ids = {"BAC009S0001W0001", "BAC009S0001W0002"}
            extra_id = (candidate_ids - {chosen_id}).pop()
            transcript = cache / "aishell_transcript_v0.8.txt"
            lines = transcript.read_text("utf-8").splitlines()
            transcript.write_text("\n".join(
                extra_id + " " + selected["dev"][0]["text"] if line.startswith(extra_id + " ") else line
                for line in lines), encoding="utf-8")
            recipe["files"][-1] = self.file_info(transcript)
            expanded = {**recipe, "train_per_speaker": 0, "frozen_holdout_recipe": recipe}
            result = choose_records(cache, expanded)
            self.assertEqual([r["utterance_id"] for r in result["train"]], [chosen_id])
            self.assertEqual(result["dev"], selected["dev"])

    def test_all_eligible_skips_empty_training_candidate_and_reports_actual_speakers(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root, include_short_candidate=True)
            build_dataset(cache, root / "base", recipe)
            expanded = {**recipe, "train_per_speaker": 0,
                        "speakers": {**recipe["speakers"], "train": ["S0004", "S0001"]},
                        "frozen_holdout_recipe": recipe}
            report = build_dataset(cache, root / "expanded", expanded)
            self.assertEqual(report["splits"]["train"]["speakers"], ["S0001"])
            self.assertEqual(report["splits"]["train"]["recordings"], 2)
            self.assertEqual(report["training_speaker_selection"], {
                "requested": ["S0004", "S0001"], "included": ["S0001"],
                "skipped_zero_eligible": ["S0004"]})
            for split in ("dev", "test"):
                self.assertEqual((root / "base" / f"{split}_full.jsonl").read_bytes(),
                                 (root / "expanded" / f"{split}_full.jsonl").read_bytes())

    def test_all_eligible_rejects_entirely_empty_training_before_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root, include_short_candidate=True)
            expanded = {**recipe, "train_per_speaker": 0,
                        "speakers": {**recipe["speakers"], "train": ["S0004"]},
                        "frozen_holdout_recipe": recipe}
            with self.assertRaisesRegex(ValueError, "train: no eligible recordings"):
                build_dataset(cache, root / "out", expanded)
            self.assertFalse((root / "out").exists())

    def test_fixed_training_count_still_rejects_empty_speaker(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root, include_short_candidate=True)
            recipe["speakers"]["train"].append("S0004")
            with self.assertRaisesRegex(ValueError, "S0004: only 0 eligible recordings; need 1"):
                build_dataset(cache, root / "out", recipe)
            self.assertFalse((root / "out").exists())

    def test_speaker_overlap_fails_before_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root)
            recipe["speakers"]["dev"] = ["S0001"]
            with self.assertRaisesRegex(ValueError, "speaker"):
                build_dataset(cache, root / "out", recipe)
            self.assertFalse((root / "out").exists())

    def test_identical_transcripts_are_excluded_across_splits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            cache, recipe = self.fixture(root)
            recipe["per_speaker"] = 2
            path = cache / "aishell_transcript_v0.8.txt"
            lines = path.read_text("utf-8").splitlines()
            # Both eligible dev texts are identical to the two training texts.
            lines = [line.replace("测 试 2", "测 试 1") for line in lines]
            path.write_text("\n".join(lines), encoding="utf-8")
            recipe["files"][-1] = self.file_info(path)
            with self.assertRaisesRegex(ValueError, "S0002: only 0 eligible"):
                build_dataset(cache, root / "out", recipe)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
