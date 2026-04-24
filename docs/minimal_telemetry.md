# Minimal Telemetry

This project keeps telemetry intentionally small for the 24h run:

- `step_name`
- `context_request_ids`
- `generation_request_ids`
- `num_context_requests`
- `num_generation_requests`
- `planned_total_tokens`
- `planned_total_pressure`
- `deferred_request_ids`
- `notes`

Telemetry is written as JSONL so we can:

- inspect scheduling behavior quickly,
- correlate a run with a specific workload,
- compare baseline and patched paths without a large tracing subsystem.
