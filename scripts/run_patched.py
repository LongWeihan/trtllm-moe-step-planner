from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.moe_microbatch_scheduler import install_patch, prime_workload_metadata
from scheduler.resource_model import build_step_plan
from scheduler.telemetry import JsonlTelemetryRecorder, NullTelemetryRecorder


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


@dataclass
class _FakeState:
    value: int


class PendingRequestProxy:
    CONTEXT_INIT = _FakeState(2)
    GENERATION_IN_PROGRESS = _FakeState(3)

    def __init__(self, record: dict):
        self.request_id = int(record["request_id"])
        self.state = self.GENERATION_IN_PROGRESS
        self.beam_width = 1
        self.num_draft_tokens = 0
        self.py_moe_pressure_class = record["pressure_class"]
        self.py_moe_pressure_score = float(record["pressure_score"])
        self.py_moe_pressure_group = record.get("pressure_group", record["pressure_class"])
        self.py_moe_metadata = record


def plan_external_step(
    pending_records: list[dict],
    *,
    microbatch_size: int,
    scheduler_max_tokens: int | None,
):
    request_map = {int(record["request_id"]): record for record in pending_records}
    proxies = [PendingRequestProxy(record) for record in pending_records]
    step_plan = build_step_plan(
        active_requests=proxies,
        inflight_request_ids=set(),
        max_batch_size=microbatch_size,
        max_num_tokens=scheduler_max_tokens,
        can_be_scheduled_fn=lambda req: True,
    )
    selected_ids = [
        int(getattr(req, "request_id", -1))
        for req in (step_plan.generation_requests + step_plan.context_requests)
    ]
    if not selected_ids:
        selected_ids = [int(pending_records[0]["request_id"])]
    selected_id_set = set(selected_ids)
    batch_records = [record for record in pending_records if int(record["request_id"]) in selected_id_set]
    remaining_records = [
        record for record in pending_records if int(record["request_id"]) not in selected_id_set
    ]
    return step_plan, batch_records, remaining_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model path or TRT-LLM engine dir")
    parser.add_argument("--backend", choices=("trt", "torch"), default="trt")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--scheduler-max-tokens", type=int, default=None)
    args = parser.parse_args()

    records = load_records(args.workload, args.limit)
    recorder = JsonlTelemetryRecorder(args.telemetry_output) if args.telemetry_output else NullTelemetryRecorder()

    if args.backend == "torch":
        prime_workload_metadata(records)
        install_patch(telemetry_output_path=args.telemetry_output)

    llm = build_llm(args)
    serialized_records: list[dict] = []
    batch_plan: list[dict] = []

    if args.backend == "torch":
        prompts = [record["prompt"] for record in records]
        sampling_params = build_sampling_params(records)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        batch_wall_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(outputs, list):
            outputs = [outputs]
        serialized_records.extend(
            serialize_output(record, output, batch_id=0, batch_wall_ms=batch_wall_ms)
            for record, output in zip(records, outputs, strict=False)
        )
        batch_plan.append(
            {
                "batch_id": 0,
                "request_ids": [int(record["request_id"]) for record in records],
                "pressure_classes": [record["pressure_class"] for record in records],
                "batch_wall_ms": batch_wall_ms,
                "scheduler_mode": "internal_patch",
            }
        )
    else:
        pending_records = list(records)
        batch_id = 0
        while pending_records:
            step_plan, batch_records, pending_records = plan_external_step(
                pending_records,
                microbatch_size=args.microbatch_size,
                scheduler_max_tokens=args.scheduler_max_tokens,
            )
            prompts = [record["prompt"] for record in batch_records]
            sampling_params = build_sampling_params(batch_records)
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            batch_wall_ms = (time.perf_counter() - start) * 1000.0
            if not isinstance(outputs, list):
                outputs = [outputs]

            recorder.record_step(
                step_name="external_pressure_plan",
                context_request_ids=[
                    int(getattr(req, "request_id", -1)) for req in step_plan.context_requests
                ],
                generation_request_ids=[
                    int(getattr(req, "request_id", -1)) for req in step_plan.generation_requests
                ],
                planned_total_tokens=step_plan.planned_total_tokens,
                planned_total_pressure=step_plan.planned_total_pressure,
                deferred_request_ids=step_plan.deferred_request_ids,
                notes={
                    **step_plan.notes,
                    "batch_id": batch_id,
                    "batch_wall_ms": batch_wall_ms,
                    "batch_request_ids": [int(record["request_id"]) for record in batch_records],
                },
            )

            batch_plan.append(
                {
                    "batch_id": batch_id,
                    "request_ids": [int(record["request_id"]) for record in batch_records],
                    "pressure_classes": [record["pressure_class"] for record in batch_records],
                    "planned_total_pressure": step_plan.planned_total_pressure,
                    "planned_total_tokens": step_plan.planned_total_tokens,
                    "deferred_request_ids": step_plan.deferred_request_ids,
                    "batch_wall_ms": batch_wall_ms,
                    "scheduler_mode": "external_resource_model",
                }
            )
            serialized_records.extend(
                serialize_output(record, output, batch_id=batch_id, batch_wall_ms=batch_wall_ms)
                for record, output in zip(batch_records, outputs, strict=False)
            )
            batch_id += 1

    payload = {
        "mode": "patched",
        "backend": args.backend,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "workload": str(args.workload),
        "telemetry_output": str(args.telemetry_output) if args.telemetry_output else None,
        "num_requests": len(records),
        "microbatch_size": args.microbatch_size,
        "scheduler_max_tokens": args.scheduler_max_tokens,
        "batch_plan": batch_plan,
        "records": serialized_records,
    }

    if hasattr(llm, "shutdown"):
        llm.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
