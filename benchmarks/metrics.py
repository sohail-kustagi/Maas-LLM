import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass
class BenchmarkResult:
    model_name: str
    scenario_count: int
    valid_output_count: int
    safety_pass_count: int
    latencies_ms: list[float]
    generated_tokens: int = 0
    model_size_bytes: int = 0
    peak_rss_bytes: int = 0

    @property
    def valid_output_rate(self) -> float:
        return self.valid_output_count / self.scenario_count if self.scenario_count else 0.0

    @property
    def safety_pass_rate(self) -> float:
        return self.safety_pass_count / self.scenario_count if self.scenario_count else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        values = sorted(self.latencies_ms)
        index = min(len(values) - 1, int(len(values) * 0.95))
        return values[index]

    @property
    def mean_latency_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0

    def as_dict(self) -> dict:
        result = asdict(self)
        result.update(
            valid_output_rate=self.valid_output_rate,
            safety_pass_rate=self.safety_pass_rate,
            mean_latency_ms=self.mean_latency_ms,
            p95_latency_ms=self.p95_latency_ms,
        )
        return result


def load_jsonl(path: Path) -> Iterable[Mapping]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run_benchmark(model_name: str, records: Iterable[Mapping], infer) -> BenchmarkResult:
    records = list(records)
    result = BenchmarkResult(
        model_name=model_name,
        scenario_count=len(records),
        valid_output_count=0,
        safety_pass_count=0,
        latencies_ms=[],
    )
    for record in records:
        started = time.perf_counter()
        output = infer(record)
        result.latencies_ms.append((time.perf_counter() - started) * 1000)
        if isinstance(output, Mapping):
            result.valid_output_count += 1
            if output.get("action") == record["expected"].get("action"):
                result.safety_pass_count += 1
    return result


def write_result(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
