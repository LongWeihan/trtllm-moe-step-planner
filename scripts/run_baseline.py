from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_records(path: Path, limit: int | None = None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit is not None:
        records = records[:limit]
    return records


def normalize_metrics(metrics_dict) -> dict:
    normalized = {}
    for key, value in (metrics_dict or {}).items():
        normalized[str(key)] = value
    return normalized


def serialize_output(record: dict, output, *, batch_id: int, batch_wall_ms: float) -> dict:
    completion = ""
    finish_reason = None
    output_token_ids = []
    if getattr(output, "outputs", None):
        first = output.outputs[0]
        completion = getattr(first, "text", "")
        finish_reason = str(getattr(first, "finish_reason", None))
        output_token_ids = list(getattr(first, "token_ids", []) or [])
    return {
        "request_id": getattr(output, "request_id", record.get("request_id")),
        "pressure_class": record.get("pressure_class"),
        "pressure_score": record.get("pressure_score"),
        "prompt": record.get("prompt"),
        "completion": completion,
        "finish_reason": finish_reason,
        "output_token_ids": output_token_ids,
        "max_tokens": int(record.get("max_tokens", 0)),
        "batch_id": batch_id,
        "batch_wall_ms": batch_wall_ms,
        "metrics_dict": normalize_metrics(getattr(output, "metrics_dict", {})),
    }


def chunk_records(records: list[dict], batch_size: int) -> list[list[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def build_llm(args):
    if args.backend == "trt":
        from tensorrt_llm._tensorrt_engine import LLM

        return LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            max_batch_size=args.microbatch_size,
        )

    from tensorrt_llm import LLM

    return LLM(model=args.model, max_batch_size=args.microbatch_size)


def build_sampling_params(batch_records: list[dict]):
    from tensorrt_llm import SamplingParams

    return [
        SamplingParams(
            max_tokens=int(record.get("max_tokens", 128)),
            return_perf_metrics=True,
        )
        for record in batch_records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model path or TRT-LLM engine dir")
    parser.add_argument("--backend", choices=("trt", "torch"), default="trt")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=4)
    args = parser.parse_args()

    records = load_records(args.workload, args.limit)
    llm = build_llm(args)

    serialized_records: list[dict] = []
    batch_plan: list[dict] = []

    for batch_id, batch_records in enumerate(chunk_records(records, args.microbatch_size)):
        prompts = [record["prompt"] for record in batch_records]
        sampling_params = build_sampling_params(batch_records)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        batch_wall_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(outputs, list):
            outputs = [outputs]

        batch_plan.append(
            {
                "batch_id": batch_id,
                "request_ids": [int(record["request_id"]) for record in batch_records],
                "pressure_classes": [record["pressure_class"] for record in batch_records],
                "batch_wall_ms": batch_wall_ms,
            }
        )

        serialized_records.extend(
            serialize_output(record, output, batch_id=batch_id, batch_wall_ms=batch_wall_ms)
            for record, output in zip(batch_records, outputs, strict=False)
        )

    payload = {
        "mode": "baseline",
        "backend": args.backend,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "workload": str(args.workload),
        "num_requests": len(records),
        "microbatch_size": args.microbatch_size,
        "batch_plan": batch_plan,
        "records": serialized_records,
    }

    if hasattr(llm, "shutdown"):
        llm.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
