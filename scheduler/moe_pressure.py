from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class PressureClass(StrEnum):
    BALANCED = "balanced"
    HOT_EXPERT = "hot_expert"
    HOT_RANK = "hot_rank"


DEFAULT_PRESSURE_SCORES: dict[PressureClass, float] = {
    PressureClass.BALANCED: 1.0,
    PressureClass.HOT_EXPERT: 2.2,
    PressureClass.HOT_RANK: 2.6,
}


@dataclass(slots=True)
class PressureSignal:
    pressure_class: PressureClass
    pressure_score: float
    group: str | None = None
    note: str | None = None


def normalize_pressure_class(value: Any) -> PressureClass:
    if isinstance(value, PressureClass):
        return value
    if value is None:
        return PressureClass.BALANCED
    text = str(value).strip().lower()
    if text == PressureClass.HOT_EXPERT:
        return PressureClass.HOT_EXPERT
    if text == PressureClass.HOT_RANK:
        return PressureClass.HOT_RANK
    return PressureClass.BALANCED


def pressure_from_metadata(metadata: Mapping[str, Any] | None) -> PressureSignal:
    metadata = metadata or {}
    pressure_class = normalize_pressure_class(metadata.get("pressure_class"))
    raw_score = metadata.get("pressure_score")
    pressure_score = (
        float(raw_score)
        if raw_score is not None
        else DEFAULT_PRESSURE_SCORES[pressure_class]
    )
    return PressureSignal(
        pressure_class=pressure_class,
        pressure_score=pressure_score,
        group=metadata.get("pressure_group"),
        note=metadata.get("note"),
    )


def pressure_from_request(req: Any) -> PressureSignal:
    direct_metadata = getattr(req, "py_moe_metadata", None)
    if isinstance(direct_metadata, Mapping):
        return pressure_from_metadata(direct_metadata)

    pressure_class = normalize_pressure_class(
        getattr(req, "py_moe_pressure_class", None)
    )
    raw_score = getattr(req, "py_moe_pressure_score", None)
    pressure_score = (
        float(raw_score)
        if raw_score is not None
        else DEFAULT_PRESSURE_SCORES[pressure_class]
    )
    return PressureSignal(
        pressure_class=pressure_class,
        pressure_score=pressure_score,
        group=getattr(req, "py_moe_pressure_group", None),
        note=getattr(req, "py_moe_note", None),
    )


def attach_request_pressure(req: Any, signal: PressureSignal) -> Any:
    setattr(req, "py_moe_pressure_class", signal.pressure_class.value)
    setattr(req, "py_moe_pressure_score", float(signal.pressure_score))
    setattr(req, "py_moe_pressure_group", signal.group)
    setattr(req, "py_moe_note", signal.note)
    setattr(
        req,
        "py_moe_metadata",
        {
            "pressure_class": signal.pressure_class.value,
            "pressure_score": signal.pressure_score,
            "pressure_group": signal.group,
            "note": signal.note,
        },
    )
    return req
