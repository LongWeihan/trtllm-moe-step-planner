from __future__ import annotations

import json


def main() -> None:
    import tensorrt_llm
    from tensorrt_llm import LLM

    print(
        json.dumps(
            {
                "tensorrt_llm_version": getattr(tensorrt_llm, "__version__", "unknown"),
                "llm_class": getattr(LLM, "__name__", "LLM"),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
