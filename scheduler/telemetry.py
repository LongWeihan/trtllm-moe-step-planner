from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StepTelemetryEvent:
    step_name: str
    timestamp: float
    context_request_ids: list[int] = field(default_factory=list)
    generation_request_ids: list[int] = field(default_factory=list)
    num_context_requests: int = 0
    num_generation_requests: int = 0
    planned_total_tokens: int = 0
    planned_total_pressure: float = 0.0
    deferred_request_ids: list[int] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


class JsonlTelemetryRecorder:
    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: StepTelemetryEvent) -> None:
        if self.output_path is None:
            return
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=True) + "\n")

    def record_step(
        self,
        *,
        step_name: str,
        context_request_ids: list[int],
        generation_request_ids: list[int],
        planned_total_tokens: int,
        planned_total_pressure: float,
        deferred_request_ids: list[int] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        self.record(
            StepTelemetryEvent(
                step_name=step_name,
                timestamp=time.time(),
                context_request_ids=context_request_ids,
                generation_request_ids=generation_request_ids,
                num_context_requests=len(context_request_ids),
                num_generation_requests=len(generation_request_ids),
                planned_total_tokens=planned_total_tokens,
                planned_total_pressure=planned_total_pressure,
                deferred_request_ids=deferred_request_ids or [],
                notes=notes or {},
            )
        )


class NullTelemetryRecorder(JsonlTelemetryRecorder):
    def __init__(self) -> None:
        super().__init__(None)
