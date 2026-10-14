import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class CachedOverfitTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        (self.run / "features").mkdir(parents=True)
        # Two recordings, each with two cumulative events. Dev has no cache:
        # any accidental dev access fails instead of leaking evaluation data.
        rows = [{"audio": name, "end_sample": end, "target_token_ids": [token]}
                for name, token in (("a.wav", 10), ("b.wav", 20))
                for end in (32000, 64000)]
        config = {"target_token_source": "recorded", "vocabulary": [None, 10, 20, 30],
                  "split": {"train": rows, "dev": [{"audio": "dev.wav"}]}}
        (self.run / "run.json").write_text(json.dumps(config))
        for i in range(4):
            feature = torch.tensor([[4., 0.]] if i < 2 else [[0., 4.]])
            torch.save({"features": feature.repeat(3, 1)}, self.run / "features" / f"{i:06d}.pt")

    def test_selects_complete_recordings_and_preserves_original_vocabulary(self):
        from overfit_cached import load_subset
        metadata, items = load_subset(self.run, recordings=1, seed=42)
        self.assertEqual(len(items), 2)
        self.assertEqual(len({x["audio"] for x in items}), 1)
        self.assertEqual(metadata["vocabulary"], [None, 10, 20, 30])
        self.assertEqual({x["end_sample"] for x in items}, {32000, 64000})

    def test_rejects_nonfinite_cache_and_unrepresentable_target(self):
        from overfit_cached import load_subset
        path = self.run / "features" / "000000.pt"
        self.torch.save({"features": self.torch.full((3, 2), float("nan"))}, path)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_subset(self.run, recordings=2, seed=42)
        self.torch.save({"features": self.torch.ones(3, 2)}, path)
        raw = json.loads((self.run / "run.json").read_text())
        raw["split"]["train"][0]["target_token_ids"] = [999]
        (self.run / "run.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            load_subset(self.run, recordings=2, seed=42)

    def test_real_ctc_optimization_memorizes_and_reports_updates(self):
        from overfit_cached import run_probe
        previous_threads = self.torch.get_num_threads()
        self.torch.set_num_threads(1)
        self.addCleanup(self.torch.set_num_threads, previous_threads)
        output = self.root / "probe"
        result = run_probe(self.run, output, "cpu", recordings=2, steps=200,
                           batch_size=4, learning_rate=0.05, eval_every=20, seed=42)
        self.assertEqual(result["status"], "memorized_subset")
        self.assertEqual(result["token_error_rate"], 0)
        self.assertGreater(result["step"], 0)
        self.assertLessEqual(result["step"], 200)
        history = [json.loads(x) for x in (output / "history.jsonl").read_text().splitlines()]
        self.assertEqual(history[0]["step"], 0)
        self.assertGreater(history[1]["gradient_norm"], 0)
        self.assertGreater(history[1]["weight_update_max_abs"], 0)
        self.assertTrue((output / "last_head.pt").is_file())
        with self.assertRaises(FileExistsError):
            run_probe(self.run, output, "cpu", recordings=2, steps=1)


if __name__ == "__main__":
    unittest.main()
