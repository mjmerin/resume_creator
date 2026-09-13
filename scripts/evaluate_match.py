#!/usr/bin/env python3
"""Compare a resume variant with a job posting using a local Ollama model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data" / "profile.yaml"
VARIANTS_DIR = ROOT / "variants"
DEFAULT_MODEL = "gemma2:9b-instruct-q8_0"
DEFAULT_HOST = "http://127.0.0.1:11434"
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "summary": {"type": "string"},
        "matched_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["requirement", "evidence"],
            },
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
        "keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "status": {"type": "string", "enum": ["present", "missing"]},
                },
                "required": ["keyword", "status"],
            },
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "score",
        "summary",
        "matched_requirements",
        "gaps",
        "keywords",
        "recommendations",
    ],
}

def load_variant(slug: str) -> str:
    path = VARIANTS_DIR / f"{slug}.yaml"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in VARIANTS_DIR.glob("*.yaml")))
        raise ValueError(f"Unknown variant {slug!r}. Available variants: {available or 'none'}")
    variant = path.read_text(encoding="utf-8")
    slug_match = re.search(r"(?m)^slug:\s*['\"]?([^\s#'\"]+)", variant)
    if slug_match is None or slug_match.group(1) != slug:
        raise ValueError(f"Variant slug in {path} does not match its filename")
    return variant


def profile_without_identity(profile: str) -> str:
    """Remove the top-level identity block before sending the profile to Ollama."""
    output = []
    skipping_identity = False
    for line in profile.splitlines():
        if re.match(r"^identity\s*:", line):
            skipping_identity = True
            continue
        if skipping_identity and re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", line):
            skipping_identity = False
        if not skipping_identity:
            output.append(line)
    redacted = "\n".join(output).strip()
    if not redacted:
        raise ValueError("Profile contains no job-relevant content")
    return redacted


def prompt_for(profile: str, variant: str, job_posting: str) -> str:
    return f"""You are a rigorous resume-to-job match analyst. Compare only the resume and job posting below.
Do not invent experience, infer unstated skills, or consider protected characteristics. Treat required qualifications as more important than preferred qualifications. Give a calibrated integer score from 0 to 100 for how closely this submitted resume matches the role: 90-100 exceptional direct fit, 75-89 strong fit with minor gaps, 55-74 partial fit with material gaps, 30-54 weak fit, and 0-29 little relevant evidence.

Evidence must quote or accurately paraphrase a specific resume fact. Gaps must be requirements not evidenced by the resume, not claims that the candidate lacks them. Recommendations must only suggest truthful tailoring, clarification, or areas to develop; never suggest fabrication. Keep each list concise (at most 8 items).

The resume source and selected variant below are YAML. Construct the submitted resume by using the variant's summary and skills, resolving each experience role and selected bullet ID against the profile, and including the profile's education and projects. Do not count unselected role bullets as resume evidence.

RESUME PROFILE (contact information removed)
---
{profile}
---

SELECTED RESUME VARIANT
---
{variant}
---

JOB POSTING
---
{job_posting}
---"""


def api_url(host: str) -> str:
    host = host.rstrip("/")
    return f"{host}/chat" if host.endswith("/api") else f"{host}/api/chat"


def call_ollama(host: str, model: str, prompt: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": REPORT_SCHEMA,
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = Request(
        api_url(host), data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start it with `ollama serve` and try again."
        ) from error
    content = body.get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama returned no chat response")
    return parse_report(content)


def parse_report(content: str) -> dict[str, Any]:
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip())
    try:
        report = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Ollama returned invalid JSON: {error.msg}") from error
    if not isinstance(report, dict):
        raise RuntimeError("Ollama returned a JSON value instead of an object")
    missing = [key for key in REPORT_SCHEMA["required"] if key not in report]
    if missing:
        raise RuntimeError(f"Ollama response is missing: {', '.join(missing)}")
    score = report["score"]
    if type(score) is not int or not 0 <= score <= 100:
        raise RuntimeError("Ollama response score must be an integer from 0 to 100")
    if not isinstance(report["summary"], str):
        raise RuntimeError("Ollama response summary must be a string")
    for key in ("matched_requirements", "gaps", "keywords", "recommendations"):
        if not isinstance(report[key], list):
            raise RuntimeError(f"Ollama response field {key!r} must be a list")
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("requirement"), str)
        and isinstance(item.get("evidence"), str)
        for item in report["matched_requirements"]
    ):
        raise RuntimeError("Ollama response has an invalid matched requirement")
    if not all(isinstance(item, str) for item in report["gaps"] + report["recommendations"]):
        raise RuntimeError("Ollama response has an invalid gap or recommendation")
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("keyword"), str)
        and item.get("status") in {"present", "missing"}
        for item in report["keywords"]
    ):
        raise RuntimeError("Ollama response has an invalid keyword")
    return report


def markdown_list(items: list[Any], empty_message: str) -> str:
    if not items:
        return f"- {empty_message}"
    lines = []
    for item in items:
        if isinstance(item, dict):
            if "requirement" in item and "evidence" in item:
                lines.append(f"- **{item['requirement']}** — {item['evidence']}")
            elif "keyword" in item and "status" in item:
                lines.append(f"- `{item['keyword']}` — {item['status']}")
        elif isinstance(item, str):
            lines.append(f"- {item}")
    return "\n".join(lines) if lines else f"- {empty_message}"


def render_report(report: dict[str, Any], variant: str, model: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""# Resume Match Report

**Match score: {report['score']}/100**

{report['summary']}

## Matched requirements

{markdown_list(report['matched_requirements'], 'No direct matches identified.')}

## Gaps to review

{markdown_list(report['gaps'], 'No gaps identified.')}

## Keywords

{markdown_list(report['keywords'], 'No keyword analysis returned.')}

## Truthful tailoring recommendations

{markdown_list(report['recommendations'], 'No recommendations returned.')}

---

Evaluated locally with `{model}` against the `{variant}` variant on {timestamp}. No résumé or job-posting content is sent to a cloud service.
"""


def read_job_posting(job_file: Path | None) -> str:
    if job_file:
        try:
            text = job_file.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"Could not read job posting file {job_file}: {error}") from error
    else:
        if sys.stdin.isatty():
            raise ValueError("Paste the job posting through standard input or pass --job-file PATH")
        text = sys.stdin.read()
    if not text.strip():
        raise ValueError("Job posting is empty")
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="general", help="Resume variant to evaluate (default: general)")
    parser.add_argument("--job-file", type=Path, help="UTF-8 text file containing the job posting")
    parser.add_argument(
        "--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL), help="Ollama model name"
    )
    parser.add_argument(
        "--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST), help="Ollama server URL"
    )
    parser.add_argument("--output", type=Path, help="Markdown report path (default: build/matches/)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PROFILE_PATH.exists():
        raise FileNotFoundError("Missing data/profile.yaml. Run `make init`, then add your résumé data.")
    job_posting = read_job_posting(args.job_file)
    variant = load_variant(args.variant)
    profile = profile_without_identity(PROFILE_PATH.read_text(encoding="utf-8"))
    report = call_ollama(args.host, args.model, prompt_for(profile, variant, job_posting))
    rendered = render_report(report, args.variant, args.model)
    output = args.output
    if output is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output = ROOT / "build" / "matches" / f"{args.variant}-{timestamp}.md"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Saved report: {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
