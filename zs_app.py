"""Entry point for the source and PyInstaller builds of the ZS application."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
UI_DIR = PROJECT_DIR / "UI"
# Rebuild the two local entries so the project root takes precedence for
# duplicated support-module names such as format_layered_psd.py.
for directory in (PROJECT_DIR, UI_DIR):
    text = str(directory)
    while text in sys.path:
        sys.path.remove(text)
sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(PROJECT_DIR))


def _restore_output_streams() -> None:
    """Connect the windowed bootloader's helper process to the parent's pipes."""
    if sys.stdout is None:
        try:
            sys.stdout = open(1, "w", buffering=1, closefd=False)
        except OSError:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        try:
            sys.stderr = open(2, "w", buffering=1, closefd=False)
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")


def main() -> int:
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--run-inference":
        _restore_output_streams()
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from inference import main as inference_main

        inference_main()
        return 0

    from zone_gui import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
