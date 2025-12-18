import os
import cv2
import numpy as np

def save_box_results(image_paths, boxes_dict, line_thickness, output_dir, areas_of_interest):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}: failed to load.")
            continue

        # Get boxes for this image
        boxes = boxes_dict.get(img_path, [])
        boxes = [np.array(b, dtype=np.int32) for b in boxes]

        # 🚨 Skip everything if no boxes
        if len(boxes) == 0:
            print(f"{img_path}: no boxes → skipping sample.")
            continue

        # Handle missing or empty AOI
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

        # === Draw bounding boxes for visualization ===
        for box in boxes:
            points = box.reshape(-1, 2)
            bx1 = np.min(points[:, 0]) - x
            by1 = np.min(points[:, 1]) - y
            bx2 = np.max(points[:, 0]) - x
            by2 = np.max(points[:, 1]) - y

            bx1 = int(np.clip(bx1, 0, w-1))
            bx2 = int(np.clip(bx2, 0, w-1))
            by1 = int(np.clip(by1, 0, h-1))
            by2 = int(np.clip(by2, 0, h-1))

            cv2.rectangle(result, (bx1, by1), (bx2, by2),
                          (0, 255, 0), int(line_thickness))

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # === Prepare YOLO dataset folder ===
        yolo_dir = os.path.join(output_dir, f"yolo_data_{img_name}")
        os.makedirs(yolo_dir, exist_ok=True)

        # Save image
        original_img_path = os.path.join(yolo_dir, f"{img_name}.png")
        cv2.imwrite(original_img_path, cropped if not has_aoi else cropped_masked)

        # === Save YOLO bounding box annotation ===
        txt_path = os.path.join(yolo_dir, f"{img_name}.txt")
        h_img, w_img = cropped.shape[:2]

        with open(txt_path, "w") as f:
            for box in boxes:
                points = box.reshape(-1, 2)
                bx1 = np.min(points[:, 0]) - x
                by1 = np.min(points[:, 1]) - y
                bx2 = np.max(points[:, 0]) - x
                by2 = np.max(points[:, 1]) - y

                bx1 = int(np.clip(bx1, 0, w_img-1))
                bx2 = int(np.clip(bx2, 0, w_img-1))
                by1 = int(np.clip(by1, 0, h_img-1))
                by2 = int(np.clip(by2, 0, h_img-1))

                cx = (bx1 + bx2) / 2 / w_img
                cy = (by1 + by2) / 2 / h_img
                bw = (bx2 - bx1) / w_img
                bh = (by2 - by1) / h_img

                if bw > 0 and bh > 0:
                    class_id = 0
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        results.append(result)

    return results
