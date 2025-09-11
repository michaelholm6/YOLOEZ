import numpy as np
import cv2

def clip_edges_to_polygon(edges, polygon_pts, small_patch_thresh=0.1):
    """
    edges: binary array (0/1) where 1 = edge pixels, 0 = background
    polygon_pts: list of (x, y) points representing the polygon
    small_patch_thresh: relative threshold for removing small black patches
    Returns: list of contours
    """

    # Ensure uint8 binary
    edges = (edges > 0).astype(np.uint8)

    # Step 1: Invert to treat "black patches" as connected components
    inv_edges = 1 - edges

    # Step 2: Label connected components (black patches)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_edges, connectivity=8)

    # Find largest black patch
    largest_area = np.max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0

    # Remove small black patches (turn them white in 'edges')
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < small_patch_thresh * largest_area:
            edges[labels == i] = 1  # make them white (edge)

    # Step 3: Create polygon-outside mask (points outside polygon = white)
    poly_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(polygon_pts, dtype=np.int32)], 1)
    outside_mask = 1 - poly_mask  # outside polygon = 1 (white)

    # XOR between edited edges and outside mask
    xor_mask = cv2.bitwise_xor(edges, outside_mask)

    # Step 4: Keep only inside polygon (outside = black)
    final_mask = cv2.bitwise_and(xor_mask, poly_mask)

    # Step 5: Find contours
    contours, _ = cv2.findContours(final_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours