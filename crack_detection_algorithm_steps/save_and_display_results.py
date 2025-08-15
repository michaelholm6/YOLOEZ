import cv2
import numpy as np

def save_and_display_results(image, contours, skeleton, line_thickness, output_path):
    result = image.copy()

    for cnt, circularity in contours:
        color = (255, 0, 0) if circularity > 0.6 else (0, 255, 0)
        cv2.drawContours(result, [cnt], -1, color, line_thickness)
        
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    kernel = np.ones((line_thickness, line_thickness), np.uint8)  # Increase size for more thickness
    thick_skeleton = cv2.dilate(skeleton_uint8, kernel, iterations=1)
    thick_skeleton = thick_skeleton > 0

    result[thick_skeleton] = (0, 0, 255)  # Red for skeleton
    
    cv2.imwrite(output_path, result)
    
    return result