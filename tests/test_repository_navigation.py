"""Non-destructive navigation checks; neither science nor historical archive tests."""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_NAVIGATION = (
    "REPOSITORY_LAYOUT.md",
    "docs/README.md",
    "docs/ROADMAP.md",
    "docs/OPEN_QUESTIONS.md",
    "docs/REPOSITORY_OPERATIONS.md",
    "docs/WORKSTREAMS.md",
    "research/README.md",
    "research-checkpoints/README.md",
    "publication/README.md",
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class RepositoryNavigationTests(unittest.TestCase):
    def test_entrypoints_exist(self):
        for name in ACTIVE_NAVIGATION:
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file(), name)

    def test_active_navigation_links_resolve(self):
        for name in ACTIVE_NAVIGATION:
            path = ROOT / name
            text = path.read_text(encoding="utf-8")
            text = re.sub(r"```.*?```", "", text, flags=re.S)
            for target in LINK.findall(text):
                url = target.strip().split()[0].strip("<>")
                parsed = urlsplit(url)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                if parsed.path.startswith("/"):
                    self.fail(f"absolute root reference: {name} -> {target}")
                resolved = (path.parent / unquote(parsed.path)).resolve()
                with self.subTest(source=name, target=target):
                    self.assertTrue(resolved.is_relative_to(ROOT.resolve()), "link escapes checkout")
                    self.assertTrue(resolved.exists(), f"broken relative link: {name} -> {target}")

    def test_reading_routes_keep_history_separate(self):
        root = (ROOT / "REPOSITORY_LAYOUT.md").read_text(encoding="utf-8")
        docs = (ROOT / "docs/README.md").read_text(encoding="utf-8")
        operations = (ROOT / "docs/REPOSITORY_OPERATIONS.md").read_text(encoding="utf-8")
        self.assertIn("REPOSITORY_OPERATIONS.md", root)
        self.assertIn("WORKSTREAMS.md", docs)
        for token in ("v0.2.0-public-snapshot", "Issue #87", "main", "research/checkpoint-20261026"):
            self.assertIn(token, operations)
        self.assertIn("INVALIDATED_IMPLEMENTATION_BUG",
                      (ROOT / "results/phase2/precision_composition/INVALIDATED_HISTORY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
