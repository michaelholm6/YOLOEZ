import cv2
import numpy as np

def measure_contour_circularity(contour_list):
    circularity_list = []
    for cnt in contour_list:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            circularity_list.append(0)  # Avoid division by zero
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity_list.append(circularity)
    return circularity_list