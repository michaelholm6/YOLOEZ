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

import sys
import traceback

from PyQt5.QtWidgets import QApplication

from YOLO_EZ import main as YOLO_EZ_main
from utils import show_error_window


def main(test_mode: bool = False) -> int:
    if test_mode:
        return 0

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    YOLO_EZ_main()

    return app.exec_()


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
