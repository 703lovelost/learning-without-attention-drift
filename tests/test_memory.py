import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from eagle3_olmo2.prepare_features import _limit_capture_memory


class CaptureMemoryTest(unittest.TestCase):
    def test_static_fraction_is_capped_to_detected_free_memory(self):
        with TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 2048,
                        "intermediate_size": 8192,
                        "num_hidden_layers": 16,
                        "vocab_size": 100352,
                        "torch_dtype": "bfloat16",
                    }
                ),
                encoding="utf-8",
            )
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    mem_get_info=lambda: (6 << 30, 8 << 30),
                )
            )
            args = [
                "--target-model-path",
                str(target),
                "--sglang-mem-fraction-static",
                "0.8",
            ]
            memory = _limit_capture_memory(args, fake_torch)
        self.assertLess(memory["sglang_mem_fraction_effective"], 0.8)
        self.assertEqual(memory["available_before_capture_bytes"], 6 << 30)

    def test_requires_cuda_for_target_capture(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            _limit_capture_memory([], fake_torch)


if __name__ == "__main__":
    unittest.main()
