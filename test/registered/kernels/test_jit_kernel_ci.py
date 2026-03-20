import os
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, suite="stage-b-kernel")
register_cuda_ci(est_time=1800, suite="nightly-kernel", nightly=True)


class TestJITKernelCI(unittest.TestCase):
    def test_jit_kernel_pytest_suite(self):
        repo_root = Path(__file__).resolve().parents[3]
        test_dir = repo_root / "python" / "sglang" / "jit_kernel" / "tests"
        cmd = [sys.executable, "-m", "pytest", "-q", str(test_dir)]
        result = subprocess.run(cmd, cwd=repo_root, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, f"jit-kernel tests failed: {cmd}")


if __name__ == "__main__":
    unittest.main()
