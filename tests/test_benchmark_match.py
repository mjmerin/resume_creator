from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_match import evaluate_model


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkMatchTests(unittest.TestCase):
    def test_fixture_contains_twenty_unique_human_labeled_cases(self):
        fixture = json.loads((ROOT / "evaluation" / "cases.json").read_text())
        case_ids = [case["id"] for case in fixture["cases"]]
        self.assertEqual(len(case_ids), 20)
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertTrue(all(case["expected_requirements"] for case in fixture["cases"]))

    def test_metrics_flag_positive_evidence_for_an_unverified_requirement(self):
        model_report = {
            "score": 100,
            "requirements": [
                {
                    "requirement": "[R1] Rust",
                    "priority": "required",
                    "evidence_status": "evidenced",
                    "evidence": "Invented Rust experience.",
                }
            ],
        }
        cases = [
            {
                "id": "unsupported",
                "posting": "Required: [R1] Rust.",
                "expected_requirements": [
                    {"id": "R1", "priority": "required", "evidence_status": "unverified"}
                ],
            }
        ]
        with (
            patch("scripts.benchmark_match.call_ollama", return_value=model_report),
            patch("scripts.benchmark_match.loaded_vram", return_value=1024),
        ):
            result = evaluate_model("test", "http://localhost:11434", {}, cases, runs=2)

        self.assertEqual(result["invented_evidence_rate"], 1.0)
        self.assertEqual(result["required_recall"], 1.0)
        self.assertEqual(result["json_success_rate"], 1.0)
        self.assertEqual(result["score_consistency"], 1.0)


if __name__ == "__main__":
    unittest.main()
