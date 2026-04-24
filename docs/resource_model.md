# Runtime Resource Model

The compact version adds a minimal planning layer between active requests and the TRT-LLM micro-batch scheduler.

## Request Profile

Each schedulable request is mapped to a `RequestProfile` with:

- `request_id`
- `state_value`
- `pressure_class`
- `pressure_score`
- `prompt_len`
- `context_remaining_length`
- `estimated_reusable_tokens`
- `beam_width`
- `num_draft_tokens`
- `is_context`
- `is_generation`

## Runtime Budget

Each scheduling iteration derives a compact `RuntimeBudget`:

- `max_batch_size`
- `max_num_tokens`
- `pressure_budget`
- `prefill_quota`
- `generation_quota`

The important design choice is that MoE pressure becomes part of the same planning contract as token and batch limits.

## Step Plan

The planner emits a `StepPlan`:

- `context_requests`
- `generation_requests`
- `planned_total_tokens`
- `planned_total_pressure`
- `deferred_request_ids`

The patched micro-batch scheduler consumes this `StepPlan` rather than scattering ad hoc MoE heuristics throughout the schedule loop.
