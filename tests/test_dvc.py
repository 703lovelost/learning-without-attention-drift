import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eagle3_olmo2.dvc import ensure_dvc_artifacts_available


class EnsureDvcArtifactsTest(unittest.TestCase):
    def test_skips_when_repository_has_no_dvc_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(ensure_dvc_artifacts_available(Path(directory)))

    def test_skips_pipeline_definition_without_lockfile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dvc.yaml").write_text("stages: {}\n", encoding="utf-8")
            self.assertFalse(ensure_dvc_artifacts_available(root))

    def test_pulls_when_dvc_refs_are_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dvc_dir = root / ".dvc"
            dvc_dir.mkdir()
            (dvc_dir / "config").touch()
            (root / "data.jsonl.dvc").touch()
            with patch("eagle3_olmo2.dvc.subprocess.run") as run:
                self.assertTrue(ensure_dvc_artifacts_available(root))

        run.assert_called_once_with(["dvc", "pull"], cwd=root, check=True)

    def test_fails_instead_of_skipping_when_refs_lack_dvc_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features.dvc").touch()
            with self.assertRaisesRegex(RuntimeError, "not initialized"):
                ensure_dvc_artifacts_available(root)

    def test_propagates_pull_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dvc_dir = root / ".dvc"
            dvc_dir.mkdir()
            (dvc_dir / "config").touch()
            (root / "features.dvc").touch()
            with patch(
                "eagle3_olmo2.dvc.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["dvc", "pull"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    ensure_dvc_artifacts_available(root)


if __name__ == "__main__":
    unittest.main()
