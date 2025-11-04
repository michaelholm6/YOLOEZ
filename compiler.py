import subprocess
import PyInstaller.__main__

def generate_exe():

    PyInstaller.__main__.run([
        "main.py",
        "--name", "crack_detection_toolset",
        "--icon", "molecule.ico",
        "--add-data", "model.yml.gz;.",
    ])

if __name__ == "__main__":
    generate_exe()