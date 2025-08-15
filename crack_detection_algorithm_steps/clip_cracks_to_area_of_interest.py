import numpy as np
import cv2

def clip_contours_with_polygon(image_shape, contours, polygon_points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 255)
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)
    clipped = cv2.bitwise_and(contour_mask, mask)
    new_contours, _ = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return new_contours