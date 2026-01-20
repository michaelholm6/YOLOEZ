import PyInstaller.__main__
import platform

def generate_exe():
    """Generate an executable from main.py with OS and architecture appended to the name."""
    
    system_name = platform.system().lower()    
    arch = platform.machine().lower()          

    exe_name = f"YOLOEZ-{system_name}-{arch}"

    print(f"Building executable: {exe_name}")

    PyInstaller.__main__.run([
        "main.py",
        "--name", exe_name,
        "--icon", "molecule.ico"
    ])

if __name__ == "__main__":
    generate_exe()
