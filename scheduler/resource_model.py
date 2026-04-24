from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .moe_pressure import PressureClass, PressureSignal, pressure_from_request


@dataclass(slots=True)
class RequestProfile:
    request_id: int
    state_value: int
    pressure_class: PressureClass
    pressure_score: float
    prompt_len: int
    context_remaining_length: int
    estimated_reusable_tokens: int
    beam_width: int
    num_draft_tokens: int
    token_cost: int
    is_context: bool
    is_generation: bool
    raw_signal: PressureSignal


@dataclass(slots=True)
class RuntimeBudget:
    max_batch_size: int
    max_num_tokens: int | None
    pressure_budget: float
    prefill_quota: int
    generation_quota: int
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepPlan:
    context_requests: list[Any]
    generation_requests: list[Any]
    profiles: dict[int, RequestProfile]
    planned_total_tokens: int
    planned_total_pressure: float
    deferred_request_ids: list[int]
    notes: dict[str, Any] = field(default_factory=dict)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _enum_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    raw_value = getattr(value, "value", value)
    return _safe_int(raw_value, default)


def _get_state_value(req: Any) -> int:
    if hasattr(req, "state_value"):
        return _safe_int(getattr(req, "state_value"), 0)
    return _enum_value(getattr(req, "state", None), 0)


def _get_generation_beam_width(req: Any) -> int:
    getter = getattr(req, "get_beam_width_by_iter", None)
    if callable(getter):
        try:
            return int(getter(for_next_iteration=False))
        except TypeError:
            return int(getter())
    return 1


def estimate_token_cost(req: Any) -> int:
    state_value = _get_state_value(req)
    context_init = _enum_value(getattr(req.__class__, "CONTEXT_INIT", None), 2)
    encoder_init = _enum_value(getattr(req.__class__, "ENCODER_INIT", None), 1)
    if state_value == encoder_init:
        return max(1, _safe_int(getattr(req, "encoder_output_len", 0), 1))
    if state_value == context_init:
        base_tokens = 0
        getter = getattr(req, "get_num_tokens", None)
        if callable(getter):
            try:
                base_tokens = _safe_int(getter(0))
            except TypeError:
                base_tokens = _safe_int(getter())
        if base_tokens <= 0:
            base_tokens = _safe_int(getattr(req, "context_remaining_length", 0), 1)
        if base_tokens <= 0:
            base_tokens = _safe_int(
                getattr(req, "py_orig_prompt_len", getattr(req, "orig_prompt_len", 0)),
                1,
            )
        reusable = (
            _safe_int(getattr(req, "estimated_reusable_tokens", 0))
            if bool(getattr(req, "is_first_context_chunk", True))
            else 0
        )
        base_tokens = max(1, base_tokens - min(reusable, base_tokens))
        return max(1, base_tokens + _safe_int(getattr(req, "num_draft_tokens", 0)))
    beam_width = _get_generation_beam_width(req)
    return max(1, beam_width + _safe_int(getattr(req, "num_draft_tokens", 0)))


def build_request_profile(req: Any) -> RequestProfile:
    signal = pressure_from_request(req)
    state_value = _get_state_value(req)
    context_init_value = _enum_value(getattr(req.__class__, "CONTEXT_INIT", None), 2)
    is_context = state_value == context_init_value
    is_generation = not is_context
    return RequestProfile(
        request_id=_safe_int(getattr(req, "request_id", -1)),
        state_value=state_value,
        pressure_class=signal.pressure_class,
        pressure_score=float(signal.pressure_score),
        prompt_len=_safe_int(
            getattr(
                req,
                "py_prompt_len",
                getattr(req, "py_orig_prompt_len", getattr(req, "prompt_len", 0)),
            )
        ),
        context_remaining_length=_safe_int(getattr(req, "context_remaining_length", 0)),
        estimated_reusable_tokens=_safe_int(getattr(req, "estimated_reusable_tokens", 0)),
        beam_width=_get_generation_beam_width(req),
        num_draft_tokens=_safe_int(getattr(req, "num_draft_tokens", 0)),
        token_cost=estimate_token_cost(req),
        is_context=is_context,
        is_generation=is_generation,
        raw_signal=signal,
    )


def build_runtime_budget(
    *,
    active_requests: list[Any],
    max_batch_size: int,
    max_num_tokens: int | None,
) -> RuntimeBudget:
    max_num_tokens = None if max_num_tokens in (None, 0) else int(max_num_tokens)
    hot_requests = 0
    for req in active_requests:
        signal = pressure_from_request(req)
        if signal.pressure_class is not PressureClass.BALANCED:
            hot_requests += 1

    pressure_budget = 3.7 if hot_requests > 0 else 4.0
    prefill_quota = max(1, max_batch_size // 2)
    generation_quota = max_batch_size
    return RuntimeBudget(
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        pressure_budget=pressure_budget,
        prefill_quota=prefill_quota,
        generation_quota=generation_quota,
        notes={"num_hot_requests": hot_requests},
    )


def build_step_plan(
    *,
    active_requests: list[Any],
    inflight_request_ids: set[int],
    max_batch_size: int,
    max_num_tokens: int | None,
    can_be_scheduled_fn,
) -> StepPlan:
    budget = build_runtime_budget(
        active_requests=active_requests,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
    )
    profiles: dict[int, RequestProfile] = {}
    generation_pool: list[tuple[Any, RequestProfile]] = []
    context_pool: list[tuple[Any, RequestProfile]] = []
    deferred_ids: list[int] = []

    for req in active_requests:
        req_id = _safe_int(getattr(req, "request_id", -1))
        if req_id in inflight_request_ids or not can_be_scheduled_fn(req):
            continue
        profile = build_request_profile(req)
        profiles[profile.request_id] = profile
        if profile.is_generation:
            generation_pool.append((req, profile))
        else:
            context_pool.append((req, profile))

    generation_pool.sort(key=lambda item: item[1].request_id)
    context_pool.sort(key=lambda item: item[1].request_id)

    planned_context: list[Any] = []
    planned_generation: list[Any] = []
    planned_pressure = 0.0
    planned_tokens = 0

    def fits(profile: RequestProfile) -> bool:
        nonlocal planned_pressure, planned_tokens
        if len(planned_context) + len(planned_generation) >= budget.max_batch_size:
            return False
        if budget.max_num_tokens is not None and planned_tokens + profile.token_cost > budget.max_num_tokens:
            return False
        if planned_pressure + profile.pressure_score > budget.pressure_budget and (
            planned_context or planned_generation
        ):
            return False
        return True

    for req, profile in generation_pool:
        if len(planned_generation) >= budget.generation_quota or not fits(profile):
            deferred_ids.append(profile.request_id)
            continue
        planned_generation.append(req)
        planned_pressure += profile.pressure_score
        planned_tokens += profile.token_cost

    for req, profile in context_pool:
        if len(planned_context) >= budget.prefill_quota or not fits(profile):
            deferred_ids.append(profile.request_id)
            continue
        planned_context.append(req)
        planned_pressure += profile.pressure_score
        planned_tokens += profile.token_cost

    return StepPlan(
        context_requests=planned_context,
        generation_requests=planned_generation,
        profiles=profiles,
        planned_total_tokens=planned_tokens,
        planned_total_pressure=planned_pressure,
        deferred_request_ids=deferred_ids,
        notes=budget.notes,
    )
