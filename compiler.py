import subprocess
import PyInstaller.__main__

def generate_exe():

    PyInstaller.__main__.run([
        "YOLO_EZ.py",
        "--name", "YOLO EZ",
        "--icon", "molecule.ico"
    ])

if __name__ == "__main__":
    generate_exe()