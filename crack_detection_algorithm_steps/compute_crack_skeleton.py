import numpy as np
import cv2
from skimage.morphology import skeletonize

def compute_skeleton(contours, shape, circularity):
    binary = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(binary, [cnt for cnt, cnt_circularity in contours if cnt_circularity < circularity], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(binary > 0)
    return skeleton