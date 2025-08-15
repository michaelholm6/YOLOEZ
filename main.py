import argparse
from crack_detection_algorithm import main as crack_detection_main

def main():
    crack_detection_main()
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e} \n")
        input("Press Enter to exit...")
    
    
    
    