"""Offline-only DOSBox-X save-state control for counterfactual rollouts."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Callable


def parse_window_ids(output: str) -> tuple[str, ...]:
    """Return numeric X11 window IDs from ``xdotool search`` output."""
    return tuple(line for line in output.splitlines() if line.isdecimal())


class X11SaveStateController:
    """Drive DOSBox-X's save/load shortcuts outside the online agent path.

    DOSBox-X exposes save states through its host-key mapper but not through a
    stable external IPC API.  The offline collector therefore runs under Xvfb
    and sends the existing host shortcuts to the exact emulator PID.  Each
    emulator receives a private state file from the native launcher.
    """

    def __init__(
        self,
        *,
        pid: int,
        state_file: str | Path,
        timeout_s: float = 5.0,
    ) -> None:
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("offline save-state branching requires an X11 DISPLAY")
        executable = shutil.which("xdotool")
        if executable is None:
            raise RuntimeError("offline save-state branching requires xdotool")
        self.pid = int(pid)
        self.state_file = Path(state_file).resolve()
        self.timeout_s = float(timeout_s)
        self._xdotool = executable
        self.window_id = self._find_window()
        self.saved_stage_frame: int | None = None

    def _find_window(self) -> str:
        deadline = time.monotonic() + self.timeout_s
        last_error = "no matching X11 window"
        while time.monotonic() < deadline:
            result = subprocess.run(
                [self._xdotool, "search", "--pid", str(self.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            windows = parse_window_ids(result.stdout)
            if windows:
                return windows[-1]
            last_error = result.stderr.strip() or last_error
            time.sleep(0.02)
        raise RuntimeError(
            f"could not find DOSBox-X X11 window for PID {self.pid}: {last_error}"
        )

    def _shortcut(self, key: str) -> None:
        self._focus()
        result = subprocess.run(
            [
                self._xdotool,
                "key",
                "--window",
                self.window_id,
                "--clearmodifiers",
                f"F12+{key}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xdotool failed to send DOSBox-X host+{key}: {result.stderr.strip()}"
            )

    def _focus(self) -> None:
        result = subprocess.run(
            [self._xdotool, "windowfocus", "--sync", self.window_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xdotool could not focus DOSBox-X: {result.stderr.strip()}"
            )

    def _wait_for_valid_archive(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                if self.state_file.stat().st_size > 0:
                    with zipfile.ZipFile(self.state_file) as archive:
                        if archive.testzip() is None:
                            return
            except (FileNotFoundError, OSError, zipfile.BadZipFile):
                pass
            time.sleep(0.01)
        raise TimeoutError(f"DOSBox-X did not finish writing {self.state_file}")

    def save_round_trip(
        self,
        *,
        resume: Callable[[], None],
        pause: Callable[[], None],
        stage_frame: Callable[[], int],
    ) -> int:
        """Save, advance, reload, and return the exact restored stage frame."""
        self.state_file.unlink(missing_ok=True)
        initial_frame = int(stage_frame())
        resume()
        try:
            self._shortcut("s")
            self._wait_for_valid_archive()
            # DOSBox-X deliberately calls GFX_LosingFocus() while saving.
            # Xvfb has no window manager to restore focus for us.
            self._focus()
            advance_deadline = time.monotonic() + self.timeout_s
            while int(stage_frame()) < initial_frame + 3:
                if time.monotonic() >= advance_deadline:
                    raise TimeoutError("stage frame did not advance after save-state creation")
                time.sleep(0.001)
        finally:
            pause()

        advanced_frame = int(stage_frame())
        resume()
        try:
            self._shortcut("l")
            load_deadline = time.monotonic() + self.timeout_s
            while True:
                restored_frame = int(stage_frame())
                if restored_frame < advanced_frame:
                    pause()
                    self.saved_stage_frame = int(stage_frame())
                    return self.saved_stage_frame
                if time.monotonic() >= load_deadline:
                    raise TimeoutError("stage frame did not rewind after save-state load")
                time.sleep(0.0005)
        finally:
            pause()

    def load(
        self,
        *,
        resume: Callable[[], None],
        pause: Callable[[], None],
        stage_frame: Callable[[], int],
    ) -> int:
        """Reload the saved state and stop on its exact native stage frame."""
        if self.saved_stage_frame is None:
            raise RuntimeError("save_round_trip() must run before load()")
        resume()
        try:
            self._shortcut("l")
            deadline = time.monotonic() + self.timeout_s
            while True:
                current = int(stage_frame())
                if current == self.saved_stage_frame:
                    pause()
                    return int(stage_frame())
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "save-state load did not expose the recorded native frame"
                    )
                time.sleep(0.0005)
        finally:
            pause()


__all__ = ["X11SaveStateController", "parse_window_ids"]
