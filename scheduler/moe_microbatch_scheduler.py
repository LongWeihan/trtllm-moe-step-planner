from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from .moe_pressure import pressure_from_metadata
from .resource_model import StepPlan, build_step_plan
from .telemetry import JsonlTelemetryRecorder, NullTelemetryRecorder


_WORKLOAD_METADATA_QUEUE: deque[dict[str, Any]] = deque()


def prime_workload_metadata(records: list[dict[str, Any]]) -> None:
    _WORKLOAD_METADATA_QUEUE.clear()
    _WORKLOAD_METADATA_QUEUE.extend(records)


def _make_recorder(output_path: str | Path | None):
    if output_path is None:
        return NullTelemetryRecorder()
    return JsonlTelemetryRecorder(output_path)


def _copy_pressure_metadata(executor_request: Any, llm_request: Any) -> Any:
    for name in (
        "py_moe_pressure_class",
        "py_moe_pressure_score",
        "py_moe_pressure_group",
        "py_moe_note",
        "py_moe_metadata",
    ):
        if hasattr(executor_request, name):
            setattr(llm_request, name, getattr(executor_request, name))
    return llm_request


def _attach_fallback_metadata(llm_request: Any) -> Any:
    if not _WORKLOAD_METADATA_QUEUE:
        return llm_request
    record = _WORKLOAD_METADATA_QUEUE.popleft()
    signal = pressure_from_metadata(record)
    setattr(llm_request, "py_moe_pressure_class", signal.pressure_class.value)
    setattr(llm_request, "py_moe_pressure_score", float(signal.pressure_score))
    setattr(llm_request, "py_moe_pressure_group", signal.group)
    setattr(llm_request, "py_moe_note", signal.note)
    setattr(llm_request, "py_moe_metadata", record)
    return llm_request


def build_patched_scheduler_class(base_cls, telemetry_output_path: str | Path | None = None):
    recorder = _make_recorder(telemetry_output_path)

    class PressureAwareMicroBatchScheduler(base_cls):
        def schedule(self, active_requests, inflight_request_ids):
            can_be_scheduled_fn = getattr(self, "_can_be_scheduled", None)
            if not callable(can_be_scheduled_fn):
                can_be_scheduled_fn = lambda req: True
            step_plan: StepPlan = build_step_plan(
                active_requests=list(active_requests),
                inflight_request_ids=inflight_request_ids,
                max_batch_size=self.max_batch_size,
                max_num_tokens=self.max_num_tokens,
                can_be_scheduled_fn=can_be_scheduled_fn,
            )
            recorder.record_step(
                step_name="patched_microbatch_schedule",
                context_request_ids=[
                    int(getattr(req, "request_id", -1))
                    for req in step_plan.context_requests
                ],
                generation_request_ids=[
                    int(getattr(req, "request_id", -1))
                    for req in step_plan.generation_requests
                ],
                planned_total_tokens=step_plan.planned_total_tokens,
                planned_total_pressure=step_plan.planned_total_pressure,
                deferred_request_ids=step_plan.deferred_request_ids,
                notes=step_plan.notes,
            )
            return step_plan.context_requests, step_plan.generation_requests

    PressureAwareMicroBatchScheduler.__name__ = "PressureAwareMicroBatchScheduler"
    return PressureAwareMicroBatchScheduler


def install_patch(telemetry_output_path: str | Path | None = None) -> None:
    import tensorrt_llm._torch.pyexecutor._util as util_module
    import tensorrt_llm._torch.pyexecutor.llm_request as llm_request_module
    import tensorrt_llm._torch.pyexecutor.scheduler as scheduler_module

    patched_cls = build_patched_scheduler_class(
        scheduler_module.BindMicroBatchScheduler,
        telemetry_output_path=telemetry_output_path,
    )
    scheduler_module.BindMicroBatchScheduler = patched_cls
    util_module.BindMicroBatchScheduler = patched_cls

    original_converter = llm_request_module.executor_request_to_llm_request

    def patched_executor_request_to_llm_request(*args, **kwargs):
        llm_request = original_converter(*args, **kwargs)
        executor_request = args[1] if len(args) > 1 else kwargs.get("executor_request")
        if executor_request is not None:
            _copy_pressure_metadata(executor_request, llm_request)
        if not hasattr(llm_request, "py_moe_pressure_class"):
            _attach_fallback_metadata(llm_request)
        return llm_request

    llm_request_module.executor_request_to_llm_request = patched_executor_request_to_llm_request
