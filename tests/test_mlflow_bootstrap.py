import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BOOTSTRAP = (
    Path(__file__).resolve().parents[1]
    / "src/eagle3_olmo2/mlflow_bootstrap/sitecustomize.py"
)


class MlflowBootstrapTest(unittest.TestCase):
    def test_tracking_uri_is_removed_from_public_tracker_config(self):
        fake_specforge = types.ModuleType("specforge")
        fake_specforge.__path__ = []
        fake_tracker = types.ModuleType("specforge.tracker")
        fake_tracker._public_config = lambda _args: {
            "mlflow_tracking_uri": "https://user:secret@example.invalid",
            "training.max_steps": 2,
        }
        fake_specforge.tracker = fake_tracker
        spec = importlib.util.spec_from_file_location("test_sitecustomize", BOOTSTRAP)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {"specforge": fake_specforge, "specforge.tracker": fake_tracker},
        ):
            spec.loader.exec_module(module)
            logged = fake_tracker._public_config(object())

        self.assertEqual(logged, {"training.max_steps": 2})


if __name__ == "__main__":
    unittest.main()
