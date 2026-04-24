from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DEFAULT_INPUTS = {
    "balanced_baseline": "results/baseline/balanced_full.json",
    "balanced_patched": "results/patched/balanced_full.json",
    "hot_expert_baseline": "results/baseline/hot_expert_full.json",
    "hot_expert_patched": "results/patched/hot_expert_full.json",
    "hot_rank_baseline": "results/baseline/hot_rank_full.json",
    "hot_rank_patched": "results/patched/hot_rank_full.json",
    "hot_expert_full24_baseline": "results/baseline/hot_expert_full24.json",
    "hot_expert_full24_patched": "results/patched/hot_expert_full24.json",
}


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def metric_value(record: dict, key: str) -> float:
    raw = record.get("metrics_dict", {}).get(key, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def summarize_payload(label: str, payload: dict) -> dict:
    records = payload.get("records", [])
    batch_plan = payload.get("batch_plan", [])
    ttft = [metric_value(record, "MetricNames.TTFT") for record in records]
    e2e = [metric_value(record, "MetricNames.E2E") for record in records]
    tpot = [metric_value(record, "MetricNames.TPOT") for record in records]
    queue_time = [metric_value(record, "MetricNames.REQUEST_QUEUE_TIME") for record in records]
    output_token_counts = [len(record.get("output_token_ids", []) or []) for record in records]
    batch_wall_ms = [float(batch.get("batch_wall_ms", 0.0)) for batch in batch_plan]
    total_batch_wall_s = sum(batch_wall_ms) / 1000.0
    total_output_tokens = sum(output_token_counts)

    return {
        "label": label,
        "mode": payload.get("mode"),
        "backend": payload.get("backend"),
        "workload": payload.get("workload"),
        "num_requests": len(records),
        "num_batches": len(batch_plan),
        "avg_requests_per_batch": (len(records) / len(batch_plan)) if batch_plan else 0.0,
        "avg_batch_ms": statistics.mean(batch_wall_ms) if batch_wall_ms else 0.0,
        "step_latency_std_ms": statistics.pstdev(batch_wall_ms) if len(batch_wall_ms) > 1 else 0.0,
        "step_latency_var_ms2": statistics.pvariance(batch_wall_ms) if len(batch_wall_ms) > 1 else 0.0,
        "total_batch_wall_s": total_batch_wall_s,
        "total_output_tokens": total_output_tokens,
        "throughput_tok_s": (total_output_tokens / total_batch_wall_s) if total_batch_wall_s > 0 else 0.0,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "ttft_p99_s": percentile(ttft, 0.99),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p90_s": percentile(e2e, 0.90),
        "e2e_p99_s": percentile(e2e, 0.99),
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p90_s": percentile(tpot, 0.90),
        "tpot_p99_s": percentile(tpot, 0.99),
        "queue_p50_s": percentile(queue_time, 0.50),
        "queue_p90_s": percentile(queue_time, 0.90),
        "queue_p99_s": percentile(queue_time, 0.99),
    }


def compare_pair(workload: str, baseline: dict, patched: dict) -> dict:
    def delta(key: str) -> float:
        return float(patched[key]) - float(baseline[key])

    def pct(key: str) -> float:
        base = float(baseline[key])
        if base == 0.0:
            return 0.0
        return delta(key) / base * 100.0

    return {
        "workload": workload,
        "baseline_label": baseline["label"],
        "patched_label": patched["label"],
        "delta_ttft_p90_s": delta("ttft_p90_s"),
        "delta_e2e_p90_s": delta("e2e_p90_s"),
        "delta_tpot_p90_s": delta("tpot_p90_s"),
        "delta_step_latency_std_ms": delta("step_latency_std_ms"),
        "delta_throughput_tok_s": delta("throughput_tok_s"),
        "pct_ttft_p90": pct("ttft_p90_s"),
        "pct_e2e_p90": pct("e2e_p90_s"),
        "pct_tpot_p90": pct("tpot_p90_s"),
        "pct_step_latency_std": pct("step_latency_std_ms"),
        "pct_throughput_tok_s": pct("throughput_tok_s"),
    }


def to_markdown(summary_by_label: dict[str, dict], comparisons: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Result Tables")
    lines.append("")
    lines.append("## Experiment Summaries")
    lines.append("")
    lines.append(
        "| Label | Requests | Batches | Avg Req/Batch | Avg Batch ms | Step Std ms | TTFT p90 s | E2E p90 s | TPOT p90 s | Throughput tok/s |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for label, summary in summary_by_label.items():
        lines.append(
            f"| {label} | {summary['num_requests']} | {summary['num_batches']} | "
            f"{summary['avg_requests_per_batch']:.2f} | {summary['avg_batch_ms']:.2f} | "
            f"{summary['step_latency_std_ms']:.2f} | {summary['ttft_p90_s']:.4f} | "
            f"{summary['e2e_p90_s']:.4f} | {summary['tpot_p90_s']:.4f} | "
            f"{summary['throughput_tok_s']:.2f} |"
        )
    lines.append("")
    lines.append("## Baseline vs Patched")
    lines.append("")
    lines.append(
        "| Workload | TTFT p90 delta s | E2E p90 delta s | TPOT p90 delta s | Step Std delta ms | Throughput delta tok/s |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in comparisons:
        lines.append(
            f"| {item['workload']} | {item['delta_ttft_p90_s']:.4f} | {item['delta_e2e_p90_s']:.4f} | "
            f"{item['delta_tpot_p90_s']:.4f} | {item['delta_step_latency_std_ms']:.2f} | "
            f"{item['delta_throughput_tok_s']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write the machine-readable summary JSON",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path to write the markdown table output",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_json = (
        args.output_json
        if args.output_json is not None
        else project_root / "results" / "compare_tables" / "summary.json"
    )
    output_md = (
        args.output_md
        if args.output_md is not None
        else project_root / "results" / "compare_tables" / "summary.md"
    )

    summary_by_label: dict[str, dict] = {}
    for label, relative_path in DEFAULT_INPUTS.items():
        path = project_root / relative_path
        summary_by_label[label] = summarize_payload(label, load_payload(path))

    comparisons = [
        compare_pair(
            workload="balanced_full",
            baseline=summary_by_label["balanced_baseline"],
            patched=summary_by_label["balanced_patched"],
        ),
        compare_pair(
            workload="hot_expert_full",
            baseline=summary_by_label["hot_expert_baseline"],
            patched=summary_by_label["hot_expert_patched"],
        ),
        compare_pair(
            workload="hot_rank_full",
            baseline=summary_by_label["hot_rank_baseline"],
            patched=summary_by_label["hot_rank_patched"],
        ),
        compare_pair(
            workload="hot_expert_full24",
            baseline=summary_by_label["hot_expert_full24_baseline"],
            patched=summary_by_label["hot_expert_full24_patched"],
        ),
    ]

    payload = {
        "summary_by_label": summary_by_label,
        "comparisons": comparisons,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(summary_by_label, comparisons), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
