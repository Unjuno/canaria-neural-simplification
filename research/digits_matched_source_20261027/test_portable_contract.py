"""Standard-library-only synthetic contract checks; never launch a runner.

Fixtures deliberately have an expired real-world deadline. They are not releases.
These tests have not been executed as part of source packaging.
"""
import copy
import pathlib
import tempfile
import unittest

import portable_contract as c

P = pathlib.Path(__file__).resolve().parent


def synthetic_release(root):
    return {
        "contract": c.CONTRACT,
        "replication_id": "replication-synthetic-test-only",
        "evidence_class": c.EVIDENCE_CLASS,
        "historical_attempt": c.HISTORICAL_ATTEMPT,
        "custody_attestation": c.CUSTODY_ATTESTATION,
        "parent_approved": True,
        "independent_review": "GO",
        "reviewer": "synthetic fixture, not a person or approval",
        "approval_evidence": "SYNTHETIC EXPIRED FIXTURE; NEVER AUTHORIZATION",
        "package_sha256": c.package_identity(root),
        "files": c.manifest_hashes(root),
        "deadline_utc": "2000-01-02T00:00:00Z",
    }


class PortableContractTests(unittest.TestCase):
    def setUp(self):
        self.release = synthetic_release(P)

    def test_fixture_only_accepted_at_synthetic_clock(self):
        self.assertEqual(c.validate_release(self.release, P, now_timestamp=0), self.release)
        with self.assertRaisesRegex(ValueError, "expired"):
            c.validate_release(self.release, P, now_timestamp=1_000_000_000)

    def test_missing_and_unknown_fields_rejected(self):
        with self.assertRaises(ValueError):
            c.validate_release({}, P, now_timestamp=0)
        self.release["ignore_registry"] = True
        with self.assertRaises(ValueError):
            c.validate_release(self.release, P, now_timestamp=0)

    def test_approval_and_review_required(self):
        for key, value in (("parent_approved", False), ("independent_review", "PENDING"),
                           ("reviewer", ""), ("approval_evidence", "")):
            row = copy.deepcopy(self.release)
            row[key] = value
            with self.assertRaises(ValueError):
                c.validate_release(row, P, now_timestamp=0)

    def test_file_and_package_mismatches_rejected(self):
        for key in ("files", "package_sha256"):
            row = copy.deepcopy(self.release)
            if key == "files":
                row[key]["runner.py"] = "0" * 64
            else:
                row[key] = "0" * 64
            with self.assertRaisesRegex(ValueError, "mismatch"):
                c.validate_release(row, P, now_timestamp=0)

    def test_historical_attempt_not_reopened(self):
        for key, value in (("historical_attempt", "OPEN"),
                           ("evidence_class", "fresh-data-confirmation"),
                           ("custody_attestation", "ALLOW_RETRY")):
            row = copy.deepcopy(self.release)
            row[key] = value
            with self.assertRaises(ValueError):
                c.validate_release(row, P, now_timestamp=0)

    def test_relative_identity_no_traversal(self):
        for value in ("../replication-escape", "replication-a/b", "historical", "replication-", "replication-A"):
            with self.assertRaises(ValueError):
                c.replication_id(value)

    def test_utc_deadline_required(self):
        for value in (None, "2030-01-01T00:00:00", "2030-01-01T00:00:00+01:00", "not-a-time"):
            with self.assertRaises(ValueError):
                c.deadline_timestamp(value)

    def test_context_fixed_layout_and_does_not_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            registry, output = c.context(root, "replication-synthetic", P)
            self.assertEqual(registry.relative_to(root).as_posix(), "custody/matched/replication-synthetic")
            self.assertEqual(output.relative_to(root).as_posix(), "artifacts/replication-synthetic")
            self.assertFalse(registry.exists())
            self.assertFalse(output.exists())

    def test_existing_workspace_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                c.context(pathlib.Path(directory) / "missing", "replication-synthetic", P)

    def test_package_cannot_hold_artifacts(self):
        with self.assertRaises(ValueError):
            c.context(P, "replication-synthetic", P)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            real = root / "real"
            real.mkdir()
            (root / "custody").symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                c.context(root, "replication-synthetic", P)

    def test_public_identity_omits_private_approval(self):
        exported = c.public_release(self.release)
        self.assertNotIn("reviewer", exported)
        self.assertNotIn("approval_evidence", exported)
        self.assertNotIn("parent_approved", exported)
        self.assertEqual(exported["historical_attempt"], c.HISTORICAL_ATTEMPT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
