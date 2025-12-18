import os
import cv2
import numpy as np

def save_segmentation_results(image_paths, contours_dict, line_thickness, output_dir, areas_of_interest):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        # Get contours for this image
        cnts = contours_dict.get(img_path, [])
        cnts = [np.array(c, dtype=np.int32) for c in cnts]

        # 🚨 Skip everything if no contours
        if len(cnts) == 0:
            print(f"{img_path}: no contours → skipping sample.")
            continue

        # Handle AOI
        area_of_interest = areas_of_interest.get(img_path, [])
        has_aoi = len(area_of_interest) > 0

        if has_aoi:
            aoi_np = np.array(area_of_interest, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(aoi_np)
            cropped = img[y:y+h, x:x+w].copy()

            mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
            shifted_aoi = aoi_np - [x, y]
            cv2.fillPoly(mask, [shifted_aoi], 255)
            cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

            result = cropped_masked.copy()
        else:
            print(f"{img_path}: no AOI found, using full image.")
            x, y, w, h = 0, 0, img.shape[1], img.shape[0]
            cropped = img.copy()
            result = img.copy()

        # === Draw contours ===
        for cnt in cnts:
            shifted_cnt = cnt - [x, y]
            cv2.drawContours(result, [shifted_cnt], -1, (0, 255, 0), line_thickness)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # === Prepare YOLO dataset folder ===
        yolo_dir = os.path.join(output_dir, f"yolo_data_{img_name}")
        os.makedirs(yolo_dir, exist_ok=True)

        # Save image
        original_img_path = os.path.join(yolo_dir, f"{img_name}.png")
        cv2.imwrite(original_img_path, cropped if not has_aoi else cropped_masked)

        # === Save YOLO segmentation annotation ===
        txt_path = os.path.join(yolo_dir, f"{img_name}.txt")
        ch, cw = cropped.shape[:2]

        with open(txt_path, "w") as f:
            for cnt in cnts:
                shifted_cnt = cnt - [x, y]
                polygon = []
                for point in shifted_cnt:
                    px, py = point[0]
                    polygon.extend([px / cw, py / ch])

                if len(polygon) < 6:
                    continue

                class_id = 0
                polygon_str = " ".join(f"{p:.6f}" for p in polygon)
                f.write(f"{class_id} {polygon_str}\n")

        results.append(result)

    return results
