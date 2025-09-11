import os
import cv2
import albumentations as A

def build_transform(aug_dict):
    """Build Albumentations transform from aug_dict flags."""
    transforms = []
    if aug_dict.get("flip"):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_dict.get("rotate"):
        transforms.append(A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5))
    if aug_dict.get("scale"):
        transforms.append(A.RandomScale(scale_limit=0.2, p=0.5))
    if aug_dict.get("color"):
        transforms.append(A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5))
    if aug_dict.get("blur"):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
    if aug_dict.get("noise"):
        transforms.append(A.GaussNoise(p=0.3))

    return A.Compose(transforms, p=1.0)


def augment_yolo_dataset(dataset_dir, aug_dict, output_dir, num_augments=1):
    """
    Augment YOLO segmentation dataset with Albumentations and save original images as well.
    
    Args:
        dataset_dir (str): Root YOLO dataset (with images/train, labels/train, images/val, labels/val).
        aug_dict (dict): Dict of augmentations to apply.
        output_dir (str): Where to save augmented dataset.
        num_augments (int): How many augmented copies per image.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform = build_transform(aug_dict)
    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    # Process both train and val splits
    for split in ["train", "val"]:
        img_in = os.path.join(dataset_dir, "images", split)
        lbl_in = os.path.join(dataset_dir, "labels", split)

        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for f in os.listdir(img_in):
            name, ext = os.path.splitext(f)
            if ext.lower() not in exts:
                continue

            img_path = os.path.join(img_in, f)
            lbl_path = os.path.join(lbl_in, name + ".txt")
            if not os.path.exists(lbl_path):
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # Load YOLO polygons
            polygons = []
            with open(lbl_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    pts = [float(x) for x in parts[1:]]
                    xy = [(pts[i]*w, pts[i+1]*h) for i in range(0, len(pts), 2)]
                    polygons.append({"label": class_id, "points": xy})

            # --- Save original image and labels ---
            orig_img_path = os.path.join(img_out, f"{name}{ext}")
            orig_lbl_path = os.path.join(lbl_out, f"{name}.txt")
            cv2.imwrite(orig_img_path, img)
            with open(orig_lbl_path, "w") as f:
                txt_lines = []
                for poly in polygons:
                    flat = []
                    for x, y in poly["points"]:
                        flat.extend([x / w, y / h])
                    if len(flat) < 6:
                        continue
                    txt_lines.append(f"{poly['label']} " + " ".join(f"{p:.6f}" for p in flat))
                f.write("\n".join(txt_lines))

            # --- Save augmented versions ---
            for aug_idx in range(num_augments):
                transformed = transform(image=img, polygons=[p['points'] for p in polygons])
                aug_img = transformed['image']
                aug_polygons = transformed['polygons']

                aug_img_name = f"{name}_aug{aug_idx}{ext}"
                aug_img_path = os.path.join(img_out, aug_img_name)
                cv2.imwrite(aug_img_path, aug_img)

                txt_lines = []
                for poly_idx, poly in enumerate(aug_polygons):
                    flat = []
                    for x, y in poly:
                        flat.extend([x / w, y / h])
                    if len(flat) < 6:
                        continue
                    txt_lines.append(f"{polygons[poly_idx]['label']} " + " ".join(f"{p:.6f}" for p in flat))
                aug_lbl_path = os.path.join(lbl_out, f"{name}_aug{aug_idx}.txt")
                with open(aug_lbl_path, "w") as f:
                    f.write("\n".join(txt_lines))
                    

def apply_augmentations(aug_dict, datasest_path, output_path):
    build_transform(aug_dict)
    augment_yolo_dataset(datasest_path, aug_dict, output_path, num_augments=1)