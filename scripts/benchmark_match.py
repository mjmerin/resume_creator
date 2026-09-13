#!/usr/bin/env python3
"""Benchmark Ollama models against human-labeled résumé matching cases."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    from .evaluate_match import DEFAULT_HOST, DEFAULT_MODEL, api_url, call_ollama, prompt_for
    from .resume_document import build_resume_sections, load_yaml
except ImportError:  # Support direct execution as scripts/benchmark_match.py.
    from evaluate_match import DEFAULT_HOST, DEFAULT_MODEL, api_url, call_ollama, prompt_for
    from resume_document import build_resume_sections, load_yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "cases.json"


def requirement_id(text: str) -> str | None:
    start = text.find("[")
    end = text.find("]", start + 1)
    if start >= 0 and end > start:
        return text[start + 1 : end].strip().upper()
    return None


def loaded_vram(host: str, model: str) -> int | None:
    endpoint = api_url(host).removesuffix("/chat") + "/ps"
    try:
        with urlopen(endpoint, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError):
        return None
    for item in payload.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            value = item.get("size_vram")
            return value if isinstance(value, int) else None
    return None


def evaluate_model(
    model: str,
    host: str,
    resume: dict[str, Any],
    cases: list[dict[str, Any]],
    runs: int,
) -> dict[str, Any]:
    attempts = 0
    successes = 0
    durations: list[float] = []
    case_scores: dict[str, list[int]] = defaultdict(list)
    expected_required = 0
    recalled_required = 0
    positive_claims = 0
    invented_claims = 0

    for case in cases:
        expected = {item["id"].upper(): item for item in case["expected_requirements"]}
        for run in range(runs):
            attempts += 1
            started = time.monotonic()
            try:
                report = call_ollama(host, model, prompt_for(resume, case["posting"]))
            except RuntimeError as error:
                print(f"{model} {case['id']} run {run + 1}: ERROR {error}")
                continue
            durations.append(time.monotonic() - started)
            successes += 1
            case_scores[case["id"]].append(report["score"])
            predicted = {
                item_id: item
                for item in report["requirements"]
                if (item_id := requirement_id(item["requirement"])) is not None
            }
            for item_id, label in expected.items():
                if label["priority"] == "required":
                    expected_required += 1
                    if item_id in predicted:
                        recalled_required += 1
            for item_id, item in predicted.items():
                if item["evidence_status"] == "unverified":
                    continue
                positive_claims += 1
                label = expected.get(item_id)
                if label is None or label["evidence_status"] == "unverified":
                    invented_claims += 1

    stable_cases = [scores for scores in case_scores.values() if len(scores) > 1]
    score_spreads = [max(scores) - min(scores) for scores in stable_cases]
    return {
        "model": model,
        "json_success_rate": successes / attempts if attempts else 0.0,
        "required_recall": recalled_required / expected_required if expected_required else 0.0,
        "invented_evidence_rate": invented_claims / positive_claims if positive_claims else 0.0,
        "score_consistency": (
            sum(1 for spread in score_spreads if spread == 0) / len(score_spreads)
            if score_spreads
            else None
        ),
        "mean_score_spread": statistics.mean(score_spreads) if score_spreads else None,
        "mean_runtime_seconds": statistics.mean(durations) if durations else None,
        "loaded_vram_bytes": loaded_vram(host, model),
        "attempts": attempts,
    }


def format_result(result: dict[str, Any]) -> str:
    consistency = result["score_consistency"]
    spread = result["mean_score_spread"]
    runtime = result["mean_runtime_seconds"]
    vram = result["loaded_vram_bytes"]
    return "\n".join(
        (
            f"## {result['model']}",
            "",
            f"- Invented-evidence rate: {result['invented_evidence_rate']:.1%}",
            f"- Required-requirement recall: {result['required_recall']:.1%}",
            f"- JSON success rate: {result['json_success_rate']:.1%}",
            (
                f"- Exact score consistency: {consistency:.1%}"
                if consistency is not None
                else "- Exact score consistency: n/a (use two or more runs)"
            ),
            (
                f"- Mean score spread: {spread:.1f} points"
                if spread is not None
                else "- Mean score spread: n/a"
            ),
            (
                f"- Mean runtime: {runtime:.2f} seconds"
                if runtime is not None
                else "- Mean runtime: n/a"
            ),
            (
                f"- Loaded VRAM reported by Ollama: {vram / (1024 ** 3):.2f} GiB"
                if vram is not None
                else "- Loaded VRAM reported by Ollama: unavailable"
            ),
            f"- Attempts: {result['attempts']}",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[DEFAULT_MODEL])
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    fixture = json.loads(args.cases.read_text(encoding="utf-8"))
    profile = load_yaml(ROOT / fixture["profile"])
    variant = load_yaml(ROOT / fixture["variant"])
    resume = build_resume_sections(profile, variant)
    cases = fixture["cases"][: args.limit]
    results = [
        evaluate_model(model, args.host, resume, cases, args.runs) for model in args.models
    ]
    report = "# Résumé Match Model Benchmark\n\n" + "\n\n".join(
        format_result(result) for result in results
    )
    report += (
        "\n\nReject a model first on invented-evidence rate, then compare recall, JSON "
        "reliability, score consistency, runtime, and memory.\n"
    )
    print(report)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        raise SystemExit(2)
