import subprocess
import PyInstaller.__main__

def generate_exe():
    """Easy function for generating an exe from main.py using PyInstaller.
    You must have PyInstaller installed in your Python environment.
    You may need to delete the 'build' and 'dist' folders created by PyInstaller
    between runs to avoid issues.
    """

    PyInstaller.__main__.run([
        "src/main.py",
        "--name", "YOLOEZ",
        "--icon", "molecule.ico"
    ])

if __name__ == "__main__":
    generate_exe()