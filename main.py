import threading
import sys
import os
from defect_detect import main as DefectDetect
import traceback
import keyboard  # pip install keyboard

def escape_listener():
    # Blocks until Escape is pressed
    keyboard.wait('esc')
    input("Escape pressed. Exiting program. Press Enter to confirm...")
    os._exit(0)  # immediately terminate Python process

def main():
    # Start listener thread
    thread = threading.Thread(target=escape_listener, daemon=True)
    thread.start()

    # Run your main program
    DefectDetect()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An error occurred:\n")
        traceback.print_exc()
        input("Press Enter to exit...")