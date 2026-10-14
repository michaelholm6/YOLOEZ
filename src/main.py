"""
YOLOEZ — A streamlined GUI workflow for training and deploying YOLO models.

Copyright (C) 2026 Michael Holm
Developed at Purdue University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import traceback

# On Linux, importing cv2 points QT_QPA_PLATFORM_PLUGIN_PATH at OpenCV's own
# bundled Qt plugins.  Those are built against a different Qt than PyQt5, so
# PyQt5 then fails to load the xcb platform plugin and the application aborts
# before any window appears.  Import cv2 here so it sets those variables now,
# then clear them, so Qt falls back to PyQt5's own plugins.  cv2 stays in
# sys.modules, so later imports elsewhere do not set them again.
import cv2  # noqa: F401

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)


def _make_output_encoding_safe():
    """Stop diagnostics from crashing on paths the console cannot encode.

    Several steps report skipped files by printing the path.  On Windows stdout
    defaults to the ANSI code page, so printing a path containing characters
    outside it raises UnicodeEncodeError and takes down the workflow over a
    diagnostic message.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # no console at all, e.g. a windowed build
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


_make_output_encoding_safe()

from PyQt5.QtCore import qInstallMessageHandler  # noqa: E402

from YOLO_EZ import main as YOLO_EZ_main  # noqa: E402
from utils import ensure_app, show_error_window  # noqa: E402

_SUPPRESSED_WARNINGS = (
    "Timers can only be used with threads started with QThread",
    "no QGuiApplication instance",
)


def _qt_message_handler(_msg_type, _context, message):
    """Suppress known-noisy Qt runtime warnings; forward everything else to stderr."""
    if any(w in message for w in _SUPPRESSED_WARNINGS):
        return
    print(message, file=sys.stderr)


qInstallMessageHandler(_qt_message_handler)


def main(test_mode: bool = False) -> int:
    """Application entry point.  Returns the QApplication exit code.

    Args:
        test_mode: When True, return immediately without starting the GUI (used by the test suite).
    """
    if test_mode:
        return 0

    # One QApplication for the whole process.  YOLO_EZ.main() reuses this
    # instance; each workflow runs its own blocking loop and returns when done,
    # so there is no event loop left to run once it comes back.  ensure_app
    # holds the reference: an unreferenced QApplication is garbage collected,
    # and the next widget construction then aborts.
    ensure_app()

    YOLO_EZ_main()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        error_msg = traceback.format_exc()
        print(error_msg)

        show_error_window(
            f"An unhandled exception occurred:\n\n{error_msg}",
            title="Unhandled Exception",
        )
        sys.exit(1)
