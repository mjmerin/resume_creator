"""Shared résumé data loading and deterministic variant resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "config" / "design.yaml"


def yaml_engine():
    from ruamel.yaml import YAML

    engine = YAML()
    engine.preserve_quotes = True
    engine.width = 1000
    return engine


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml_engine().load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def dump_yaml(document: dict[str, Any], stream: Any) -> None:
    yaml_engine().dump(document, stream)


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


def build_resume_sections(
    profile: dict[str, Any], variant: dict[str, Any]
) -> dict[str, Any]:
    """Resolve exactly the résumé sections selected by a variant."""
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
    return sections
