import threading
import sys
import os
from YOLO_EZ import main as YOLO_EZ_main
import traceback
import keyboard
from utils import show_error_window

def on_escape():
    print("Escape pressed. Exiting program...")
    os._exit(0)
    
keyboard.add_hotkey('esc', on_escape)

def main():
    YOLO_EZ_main()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_msg = traceback.format_exc()
        print("An unhandled exception occurred:\n", error_msg)
        show_error_window(f"An unhandled exception occurred:\n\n{error_msg}", title="Unhandled Exception")
        sys.exit(1)