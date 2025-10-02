import threading
import sys
import os
from defect_detect import main as DefectDetect
import traceback
import keyboard  # pip install keyboard

def on_escape():
    print("Escape pressed. Exiting program...")
    os._exit(0)
    
keyboard.add_hotkey('esc', on_escape)

def main():
    DefectDetect()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An error occurred:\n")
        traceback.print_exc()
        input("Press Enter to exit...")