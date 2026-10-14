# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Michael Holm
# Developed at Purdue University

import os
import albumentations as A
from utils import read_image, write_image


def build_transform(aug_dict, task):
    """Build an Albumentations Compose pipeline from a dict of enabled augmentation flags.

    Args:
        aug_dict: Keys are augmentation names (flip, rotate, scale, color, blur, noise);
                  values are booleans indicating whether each is enabled.
        task: "detection" uses BboxParams; anything else uses KeypointParams for segmentation.

    Returns:
        An Albumentations Compose transform, or NoOp if no augmentations are selected.
    """
    transforms = []

    if aug_dict.get("flip"):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_dict.get("rotate") or aug_dict.get("scale"):
        transforms.append(
            A.Affine(
                rotate=(-30, 30) if aug_dict.get("rotate") else 0,
                scale=(0.8, 1.2) if aug_dict.get("scale") else 1.0,
                translate_percent=(0.0, 0.0),
                fit_output=False,
                p=0.5,
            )
        )

    if aug_dict.get("color"):
        transforms.append(A.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.5))

    if aug_dict.get("blur"):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    if aug_dict.get("noise"):
        transforms.append(A.GaussNoise(p=0.3))

    if not transforms:
        return A.NoOp()

    if task == "detection":
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                clip=True,
                min_visibility=0.1,
            ),
        )
    else:
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )


def detect_label_format(lbl_path):
    """Return 'detection', 'segmentation', or None for an empty/unreadable label file.

    Every line is considered, not just the first: a file is only "detection" if
    all of its lines are 5-field boxes.  The training dialog uses this same
    function, so the task it locks onto cannot disagree with what the
    augmentation step decides and silently skip the image.
    """
    try:
        with open(lbl_path, "r") as f:
            lines = [line.split() for line in f if line.strip()]
    except OSError:
        return None

    lines = [parts for parts in lines if len(parts) >= 5]

    if not lines:
        return None
    if all(len(parts) == 5 for parts in lines):
        return "detection"
    return "segmentation"


def augment_yolo_dataset(
    dataset_dir, aug_dict, output_dir, num_augments=1, task="segmentation"
):
    """Copy dataset_dir to output_dir and append num_augments augmented variants of each image.

    Args:
        dataset_dir: Root of a YOLO dataset with images/{train,val} and labels/{train,val} subdirs.
        aug_dict: Augmentation flags; passed to build_transform.
        output_dir: Destination root (same structure as dataset_dir).
        num_augments: Number of augmented copies to produce per original image.
        task: "detection" or "segmentation"; controls which label format is transformed.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    transform = build_transform(aug_dict, task)

    for split in ["train", "val"]:
        img_in = os.path.join(dataset_dir, "images", split)
        lbl_in = os.path.join(dataset_dir, "labels", split)
        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        if not os.path.isdir(img_in) or not os.path.isdir(lbl_in):
            print(f"Skipping {split}: {img_in} or {lbl_in} does not exist.")
            continue

        for fname in os.listdir(img_in):
            name, ext = os.path.splitext(fname)
            if ext.lower() not in exts:
                continue

            img_path = os.path.join(img_in, fname)
            lbl_path = os.path.join(lbl_in, name + ".txt")
            if not os.path.exists(lbl_path):
                continue

            detected = detect_label_format(lbl_path)
            if detected != task and detected is not None:
                print(f"Skipping {name}: expected {task}, found {detected}")
                continue

            img = read_image(img_path)
            if img is None:
                print(f"Skipping {fname}: could not be read.")
                continue
            h, w = img.shape[:2]

            labels = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # blank or malformed line
                    try:
                        coords = [float(x) for x in parts[1:]]
                        label = int(parts[0])
                    except ValueError:
                        print(f"Skipping a malformed label line in {lbl_path}")
                        continue

                    # 4 values is a box; a polygon needs x/y pairs.  An odd
                    # count would silently become a 2-point polygon, which
                    # reads back as a bounding box.
                    if len(coords) != 4 and len(coords) % 2 != 0:
                        print(f"Skipping a label line with odd coords in {lbl_path}")
                        continue

                    labels.append({"label": label, "coords": coords})

            write_image(os.path.join(img_out, fname), img)
            with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
                for l in labels:
                    f.write(
                        f"{l['label']} "
                        + " ".join(f"{c:.6f}" for c in l["coords"])
                        + "\n"
                    )

            for k in range(num_augments):
                if task == "detection":
                    bboxes, class_labels = [], []

                    for l in labels:
                        xc, yc, bw, bh = l["coords"]
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        bboxes.append([x1, y1, x2, y2])
                        class_labels.append(l["label"])

                    out = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = out["image"]
                    aug_h, aug_w = aug_img.shape[:2]

                    aug_labels = []
                    for (x1, y1, x2, y2), cls in zip(
                        out["bboxes"], out["class_labels"]
                    ):
                        xc = ((x1 + x2) / 2) / aug_w
                        yc = ((y1 + y2) / 2) / aug_h
                        bw = (x2 - x1) / aug_w
                        bh = (y2 - y1) / aug_h
                        aug_labels.append({"label": cls, "coords": [xc, yc, bw, bh]})

                else:  # segmentation
                    polygons = []
                    for l in labels:
                        # len - 1: an odd coordinate count would otherwise run
                        # off the end of the list when pairing x with y.
                        pts = [
                            (l["coords"][i] * w, l["coords"][i + 1] * h)
                            for i in range(0, len(l["coords"]) - 1, 2)
                        ]
                        polygons.append(pts)

                    flat_pts = [pt for poly in polygons for pt in poly]
                    out = transform(image=img, keypoints=flat_pts)
                    aug_img = out["image"]
                    aug_h, aug_w = aug_img.shape[:2]

                    aug_labels = []
                    idx = 0
                    for l, poly in zip(labels, polygons):
                        n = len(poly)
                        pts = out["keypoints"][idx : idx + n]
                        idx += n

                        coords = []
                        for x, y in pts:
                            coords.extend(
                                [
                                    max(0.0, min(1.0, x / aug_w)),
                                    max(0.0, min(1.0, y / aug_h)),
                                ]
                            )

                        aug_labels.append({"label": l["label"], "coords": coords})

                aug_img_name = f"{name}_aug{k}{ext}"
                write_image(os.path.join(img_out, aug_img_name), aug_img)

                with open(os.path.join(lbl_out, f"{name}_aug{k}.txt"), "w") as f:
                    for l in aug_labels:
                        f.write(
                            f"{l['label']} "
                            + " ".join(f"{c:.6f}" for c in l["coords"])
                            + "\n"
                        )
