import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.metrics import load_jsonl, run_benchmark
from scripts.generate_sitl_dataset import generate


class DatasetAndBenchmarkTests(unittest.TestCase):
    def test_dataset_is_deterministic_and_split(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(Path(first), records_per_split=5, seed=42)
            generate(Path(second), records_per_split=5, seed=42)
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    Path(first, f"{split}.jsonl").read_text(),
                    Path(second, f"{split}.jsonl").read_text(),
                )
                records = list(load_jsonl(Path(first, f"{split}.jsonl")))
                self.assertEqual(len(records), 5)
                self.assertTrue(all(record["split"] == split for record in records))

    def test_benchmark_records_latency_and_rates(self):
        records = [{"expected": {"action": "NO_ACTION"}}]
        result = run_benchmark("test", records, lambda _record: {"action": "NO_ACTION"})
        self.assertEqual(result.scenario_count, 1)
        self.assertEqual(result.valid_output_rate, 1.0)
        self.assertEqual(result.safety_pass_rate, 1.0)
        self.assertGreaterEqual(result.mean_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
