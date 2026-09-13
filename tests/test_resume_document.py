from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.build import build_document
from scripts.resume_document import build_resume_sections


class ResumeDocumentTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "identity": {"name": "Test Person", "email": "secret@example.com", "phone": "+1"},
            "roles": {
                "engineer": {
                    "company": "Example",
                    "position": "Engineer",
                    "bullets": {
                        "selected": "Built the selected system.",
                        "unselected": "Secret unselected achievement.",
                    },
                }
            },
            "education": {"institution": "Example U", "degree": "BS", "area": "CS"},
            "projects": {},
        }
        self.variant = {
            "slug": "test",
            "summary": "Engineer summary.",
            "experience": [{"role": "engineer", "bullets": ["selected"]}],
            "skills": [{"label": "Languages", "details": "Python"}],
        }

    def test_matcher_sections_are_the_same_sections_rendered(self):
        sections = build_resume_sections(self.profile, self.variant)
        with patch("scripts.build.load_yaml", return_value={"design": {}}):
            rendered = build_document(self.profile, self.variant)
        self.assertEqual(sections, rendered["cv"]["sections"])

    def test_only_selected_bullets_are_resolved_and_identity_is_absent(self):
        serialized = str(build_resume_sections(self.profile, self.variant))

        self.assertIn("Built the selected system", serialized)
        self.assertNotIn("Secret unselected achievement", serialized)
        self.assertNotIn("secret@example.com", serialized)


if __name__ == "__main__":
    unittest.main()
