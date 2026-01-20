import sys
import traceback
from PyQt5.QtWidgets import QApplication
from YOLO_EZ import main as YOLO_EZ_main
from utils import show_error_window
from PyQt5.QtCore import QObject, QEvent, Qt
from PyQt5.QtWidgets import QApplication
import sys

class GlobalEscapeFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            print("Global ESC pressed — quitting application")
            QApplication.quit()
            return True
        return False

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.installEventFilter(GlobalEscapeFilter())

    YOLO_EZ_main()  # creates ALL windows, dialogs, etc.

    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_msg = traceback.format_exc()
        print(error_msg)
        show_error_window(
            f"An unhandled exception occurred:\n\n{error_msg}",
            title="Unhandled Exception"
        )
        sys.exit(1)
