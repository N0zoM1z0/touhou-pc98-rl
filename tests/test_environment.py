import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pc98rl.env import describe_th05_scenario, resolve_dosbox_executable


class EmulatorResolutionTest(unittest.TestCase):
    def test_scenario_uses_human_stage_number(self):
        observation = np.zeros(273, dtype=np.float32)
        observation[8] = 0.5
        observation[12] = 2.0 / 3.0
        observation[13] = 1.0 / 6.0
        observation[14] = 3.0 / 8.0
        observation[15] = 2.0 / 8.0
        observation[16] = 1.0
        self.assertEqual(
            describe_th05_scenario(observation),
            {
                "stage": 2,
                "patch_stage_index": 1,
                "character": 2,
                "rank": 3,
                "initial_power": 64,
                "configured_lives": 3,
                "configured_bombs": 2,
            },
        )

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
