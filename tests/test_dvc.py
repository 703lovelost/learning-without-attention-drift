import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eagle3_olmo2.data import sha256_file
from eagle3_olmo2.dvc import ensure_dvc_artifacts_available, register_artifact


class EnsureDvcArtifactsTest(unittest.TestCase):
    def test_uninitialized_workspace_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "features"
            artifact.mkdir()
            with self.assertRaisesRegex(RuntimeError, "not initialized"):
                ensure_dvc_artifacts_available(artifact)

    def test_ref_must_belong_to_the_requested_stage_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".dvc").mkdir()
            (root / ".dvc/config").touch()
            (root / "unrelated.dvc").touch()
            (root / "dvc.yaml").write_text("stages: {}\n")
            artifact = root / "train"
            artifact.mkdir()
            with self.assertRaisesRegex(RuntimeError, "reference is missing"):
                ensure_dvc_artifacts_available(artifact)

    @unittest.skipUnless(shutil.which("dvc"), "DVC requires the pipeline extra")
    def test_external_workspace_roundtrip_and_missing_remote_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "DVC_CACHE_DIR": str(root / "cache"),
                "DVC_REMOTE_DIR": str(root / "remote"),
                "DVC_NO_ANALYTICS": "true",
            }
            artifact = root / "external" / "train"
            artifact.mkdir(parents=True)
            sample = artifact / "record.jsonl"
            sample.write_text(
                '{"conversations":[{"role":"assistant","content":"test"}]}\n'
            )
            digest = sha256_file(sample)

            with patch.dict(os.environ, environment):
                register_artifact(artifact)
                shutil.rmtree(artifact)
                shutil.rmtree(root / "cache")
                (root / "cache").mkdir()
                ensure_dvc_artifacts_available(artifact)
                self.assertEqual(sha256_file(sample), digest)

                # A populated cache/worktree must not satisfy an empty remote.
                shutil.rmtree(root / "remote")
                (root / "remote").mkdir()
                with self.assertRaisesRegex(RuntimeError, "DVC"):
                    ensure_dvc_artifacts_available(artifact)
