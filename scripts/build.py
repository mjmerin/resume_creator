#!/usr/bin/env python3
"""Generate complete RenderCV YAML inputs and render selected resume variants."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data" / "profile.yaml"
DESIGN_PATH = ROOT / "config" / "design.yaml"
VARIANTS_DIR = ROOT / "variants"
GENERATED_DIR = ROOT / "build" / "rendercv"
TYPST_DIR = ROOT / "build" / "typst"
PREVIEW_DIR = ROOT / "build" / "previews"
DIST_DIR = ROOT / "dist"
SCHEMA_URL = "https://raw.githubusercontent.com/rendercv/rendercv/refs/tags/v2.8/schema.json"

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 1000


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def available_variants() -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for path in sorted(VARIANTS_DIR.glob("*.yaml")):
        definition = load_yaml(path)
        slug = definition.get("slug")
        if not isinstance(slug, str) or not slug:
            raise ValueError(f"Missing slug in {path}")
        if slug in variants:
            raise ValueError(f"Duplicate variant slug: {slug}")
        variants[slug] = path
    return variants


def filename_component(value: str) -> str:
    """Return a filesystem-friendly representation of a name or variant slug."""
    component = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not component:
        raise ValueError(f"Cannot create a filename from {value!r}")
    return component


def output_name(profile: dict[str, Any], variant: dict[str, Any]) -> str:
    configured_name = variant.get("output_name")
    if configured_name:
        if not isinstance(configured_name, str):
            raise ValueError(f"output_name in {variant['slug']} must be a string")
        return filename_component(configured_name)
    return "_".join(
        (
            filename_component(profile["identity"]["name"]),
            filename_component(variant["slug"]),
        )
    )


def select_highlights(
    role_id: str, role: dict[str, Any], bullet_ids: list[str]
) -> list[str]:
    bullet_bank = role.get("bullets", {})
    missing = [bullet_id for bullet_id in bullet_ids if bullet_id not in bullet_bank]
    if missing:
        raise ValueError(f"Unknown bullet(s) for {role_id}: {', '.join(missing)}")
    return [bullet_bank[bullet_id] for bullet_id in bullet_ids]


def build_experience_entry(
    role_id: str, role: dict[str, Any], bullet_ids: list[str]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "company": role["company"],
        "position": role["position"],
    }
    for key in ("date", "start_date", "end_date", "location"):
        if role.get(key):
            entry[key] = role[key]
    entry["highlights"] = select_highlights(role_id, role, bullet_ids)
    return entry


def build_document(profile: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    identity = profile["identity"]
    roles = profile["roles"]
    experiences = []
    for selection in variant["experience"]:
        role_id = selection["role"]
        if role_id not in roles:
            raise ValueError(f"Unknown role in {variant['slug']}: {role_id}")
        experiences.append(
            build_experience_entry(role_id, roles[role_id], list(selection["bullets"]))
        )

    education = profile["education"]
    projects = profile.get("projects", {})
    design = load_yaml(DESIGN_PATH)["design"]
    generated_name = output_name(profile, variant)

    project_entries = []
    for project in projects.values():
        entry: dict[str, Any] = {"name": project["name"]}
        if project.get("summary"):
            entry["summary"] = project["summary"]
        if project.get("bullets"):
            entry["highlights"] = list(project["bullets"].values())
        project_entries.append(entry)

    sections: dict[str, Any] = {
        "Professional Summary": [variant["summary"]],
        "Employment History": experiences,
        "Education": [
            {
                "institution": education["institution"],
                "degree": education["degree"],
                "area": education["area"],
            }
        ],
        "Skills": [
            {"label": skill["label"], "details": skill["details"]}
            for skill in variant["skills"]
        ],
    }
    if project_entries:
        sections["Personal Projects"] = project_entries

    return {
        "cv": {
            "name": identity["name"],
            "email": identity["email"],
            "phone": identity["phone"],
            "sections": sections,
        },
        "design": design,
        "locale": {"language": "english", "present": "Present"},
        "settings": {
            "render_command": {
                "output_folder": str(DIST_DIR),
                "typst_path": str(TYPST_DIR / f"{generated_name}.typ"),
                "pdf_path": str(DIST_DIR / f"{generated_name}.pdf"),
                "markdown_path": str(ROOT / "build" / "markdown" / f"{generated_name}.md"),
                "html_path": str(ROOT / "build" / "html" / f"{generated_name}.html"),
                "png_path": str(PREVIEW_DIR / f"{generated_name}.png"),
                "dont_generate_markdown": True,
                "dont_generate_html": True,
                "dont_generate_typst": False,
                "dont_generate_pdf": False,
                "dont_generate_png": False,
            },
            "pdf_title": variant.get(
                "pdf_title",
                f"{identity['name']} - {variant['slug'].replace('-', ' ').title()}",
            ),
        },
    }


def write_generated_yaml(document: dict[str, Any], output_name: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{output_name}_CV.yaml"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"# yaml-language-server: $schema={SCHEMA_URL}\n")
        stream.write("# Generated by scripts/build.py. Edit data/ or variants/ instead.\n")
        yaml.dump(document, stream)
    return path


def render(path: Path) -> None:
    rendercv = Path(sys.executable).with_name("rendercv")
    if not rendercv.exists():
        raise RuntimeError("RenderCV is not installed. Run `uv sync --locked` first.")
    subprocess.run([str(rendercv), "render", str(path), "--quiet"], cwd=ROOT, check=True)


def verify_pdf(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Render did not create {path}")
    page_count = len(PdfReader(path).pages)
    if page_count > 2:
        raise RuntimeError(f"{path.name} is {page_count} pages; the limit is two")
    return page_count


def clean() -> None:
    for path in (ROOT / "build", DIST_DIR):
        if path.exists() and path.parent == ROOT:
            shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", help="Variant slugs; defaults to all variants")
    parser.add_argument("--no-render", action="store_true", help="Only generate RenderCV YAML")
    parser.add_argument("--clean", action="store_true", help="Remove build/ and dist/ first")
    parser.add_argument("--clean-only", action="store_true", help="Remove build/ and dist/, then exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clean_only:
        clean()
        return 0
    if args.clean:
        clean()

    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            "Missing data/profile.yaml. Run `make init`, then replace the sample details."
        )

    paths = available_variants()
    selected = args.variants or list(paths)
    unknown = sorted(set(selected) - set(paths))
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {', '.join(unknown)}. Available: {', '.join(paths)}"
        )

    profile = load_yaml(PROFILE_PATH)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for slug in selected:
        variant = load_yaml(paths[slug])
        document = build_document(profile, variant)
        generated_name = output_name(profile, variant)
        generated_path = write_generated_yaml(document, generated_name)
        if args.no_render:
            print(f"generated {generated_path.relative_to(ROOT)}")
            continue
        render(generated_path)
        pdf_path = DIST_DIR / f"{generated_name}.pdf"
        pages = verify_pdf(pdf_path)
        print(f"built {pdf_path.relative_to(ROOT)} ({pages} page{'s' if pages != 1 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
