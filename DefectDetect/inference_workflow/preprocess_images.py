import cv2
import numpy as np
from copy import deepcopy
from DefectDetect.labelling_workflow.area_of_interest_marking import get_polygon_from_user

def preprocess_images(image_paths):
    """
    Preprocess a list of images for YOLO inference:
      1. Let the user optionally crop to an area of interest (polygon).
      2. Convert the cropped image to grayscale.

    Args:
        image_paths (list of str): List of image file paths.

    Returns:
        List of preprocessed images (numpy arrays).
    """
    preprocessed = []
    original_cropped = []

    for path in image_paths:
        # --- Load image ---
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load image: {path}")
            continue

        # --- Let user choose polygon ---
        polygon_points = get_polygon_from_user(deepcopy(img))  # returns list of [x, y]

        # --- Create a mask from polygon ---
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # --- Apply mask to image ---
        cropped = cv2.bitwise_and(img, img, mask=mask)

        # --- Optional: crop to bounding rectangle of polygon ---
        x, y, w, h = cv2.boundingRect(pts)
        cropped = cropped[y:y+h, x:x+w]

        # --- Convert to grayscale ---
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        preprocessed.append(gray)
        original_cropped.append(cropped)

    return original_cropped, preprocessed