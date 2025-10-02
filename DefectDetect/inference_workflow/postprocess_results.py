import os
import cv2

def postprocess_results(results, output_dir=None, save_images=True, print_summary=True):
    """
    Postprocess YOLO inference results.
    
    Args:
        results (list): List of ultralytics YOLO result objects.
        output_dir (str, optional): Directory to save annotated images. If None, nothing saved.
        save_images (bool): Whether to save annotated images.
        print_summary (bool): Whether to print detections to console.
    
    Returns:
        list[dict]: Structured list of detections with class, confidence, and bounding boxes.
    """
    detections_summary = []

    for i, result in enumerate(results):
        # Each result corresponds to one image
        img_detections = []

        boxes = result.boxes  # YOLO Boxes object
        names = result.names  # class names mapping (dict)

        if print_summary:
            print(f"\nImage {i+1}: {len(boxes)} detections")

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            label = names[cls_id] if names and cls_id in names else str(cls_id)

            det = {
                "class_id": cls_id,
                "class_name": label,
                "confidence": conf,
                "bbox": xyxy
            }
            img_detections.append(det)

            if print_summary:
                print(f" - {label} ({conf:.2f}) at {xyxy}")

        detections_summary.append(img_detections)

        # Optionally save annotated images
        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"result_{i+1}.jpg")

            # Use YOLO's built-in plotting function
            annotated_img = result.plot()  # returns numpy array (BGR)
            cv2.imwrite(save_path, annotated_img)

            if print_summary:
                print(f"Saved annotated result to {save_path}")

    return detections_summary