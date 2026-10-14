# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import PyInstaller.__main__
import os
import platform


def generate_exe():
    """Generate an executable from main.py with OS and architecture appended to the name."""

    system_name = platform.system().lower()
    arch = platform.machine().lower()

    exe_name = f"YOLOEZ-{system_name}-{arch}"

    print(f"Building executable: {exe_name}")

    # Resolve inputs relative to this file so the documented
    # "python src/compiler.py" works from the repository root.
    here = os.path.dirname(os.path.abspath(__file__))
    entry_point = os.path.join(here, "main.py")
    icon = os.path.join(here, os.pardir, "assets", "molecule.ico")

    PyInstaller.__main__.run([entry_point, "--name", exe_name, "--icon", icon])


if __name__ == "__main__":
    generate_exe()
