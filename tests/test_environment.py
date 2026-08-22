import os
import tempfile
import unittest
from pathlib import Path

from pc98rl.env import resolve_dosbox_executable


class EmulatorResolutionTest(unittest.TestCase):
    def test_explicit_executable_does_not_depend_on_path(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "dosbox-x"
            executable.touch(mode=0o755)
            os.chmod(executable, 0o755)
            self.assertEqual(resolve_dosbox_executable(executable), executable.resolve())

    def test_rejects_non_executable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "dosbox-x"
            executable.touch(mode=0o644)
            with self.assertRaisesRegex(FileNotFoundError, "not runnable"):
                resolve_dosbox_executable(executable)


if __name__ == "__main__":
    unittest.main()
