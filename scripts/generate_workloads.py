from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.moe_pressure import DEFAULT_PRESSURE_SCORES, PressureClass


BASE_PROMPTS = {
    PressureClass.BALANCED: "Summarize the architectural trade-offs of mixture-of-experts serving in two concise paragraphs.",
    PressureClass.HOT_EXPERT: "Analyze why a router might repeatedly concentrate tokens into a small expert subset during interactive decode.",
    PressureClass.HOT_RANK: "Explain how expert placement skew can amplify rank-level stragglers during batched generation.",
}


def build_record(index: int, pressure_class: PressureClass, max_tokens: int, group: str) -> dict:
    return {
        "request_id": index,
        "prompt": BASE_PROMPTS[pressure_class],
        "max_tokens": max_tokens,
        "pressure_class": pressure_class.value,
        "pressure_score": DEFAULT_PRESSURE_SCORES[pressure_class],
        "pressure_group": group,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def generate_balanced(count: int) -> list[dict]:
    return [
        build_record(i, PressureClass.BALANCED, 96 + (i % 3) * 16, "balanced")
        for i in range(count)
    ]


def generate_hot_expert(count: int) -> list[dict]:
    return [
        build_record(i, PressureClass.HOT_EXPERT, 128 + (i % 2) * 32, "expert_a")
        for i in range(count)
    ]


def generate_hot_rank(count: int) -> list[dict]:
    return [
        build_record(i, PressureClass.HOT_RANK, 128 + (i % 4) * 16, "rank_0")
        for i in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir
    write_jsonl(output_dir / "balanced_moe.jsonl", generate_balanced(args.count))
    write_jsonl(output_dir / "hot_expert.jsonl", generate_hot_expert(args.count))
    write_jsonl(output_dir / "hot_rank.jsonl", generate_hot_rank(args.count))


if __name__ == "__main__":
    main()
