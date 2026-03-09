import os
import shutil
import random
from pathlib import Path
from PIL import Image

SOURCE_ROOT = "../crop-dataset-eitan-already-cropped"
IMAGES_DIR = os.path.join(SOURCE_ROOT, "images")
LABELS_DIR = os.path.join(SOURCE_ROOT, "labels")

OUTPUT_ROOT = "crop-dataset-eitan-yolo"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_ROOT, "labels")

CLASS_ID_MAP = {
    1: 0
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_image_for_label(label_stem):
    for ext in IMAGE_EXTS:
        p = os.path.join(IMAGES_DIR, label_stem + ext)
        if os.path.exists(p):
            return p
    for name in os.listdir(IMAGES_DIR):
        p = os.path.join(IMAGES_DIR, name)
        if Path(name).stem == label_stem and Path(name).suffix.lower() in IMAGE_EXTS:
            return p
    return None

def convert_bbox_to_yolo(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return xc, yc, bw, bh

def convert_keypoints_to_yolo(points, w, h):
    out = []
    for i in range(0, len(points), 2):
        x = points[i] / w
        y = points[i + 1] / h
        v = 2
        out.extend([x, y, v])
    return out

def clamp01(x):
    return max(0.0, min(1.0, x))

def convert_one(label_path, split_name):
    stem = Path(label_path).stem
    image_path = find_image_for_label(stem)
    if image_path is None:
        print(f"missing image for {label_path}")
        return False

    with Image.open(image_path) as img:
        w, h = img.size

    with open(label_path, "r", encoding="utf-8") as f:
        parts = f.read().strip().split()

    if len(parts) != 13:
        print(f"bad label format: {label_path}")
        return False

    src_class = int(float(parts[0]))
    dst_class = CLASS_ID_MAP.get(src_class, src_class)

    x1, y1, x2, y2 = map(float, parts[1:5])
    kp = list(map(float, parts[5:]))

    xc, yc, bw, bh = convert_bbox_to_yolo(x1, y1, x2, y2, w, h)
    kps = convert_keypoints_to_yolo(kp, w, h)

    xc = clamp01(xc)
    yc = clamp01(yc)
    bw = clamp01(bw)
    bh = clamp01(bh)
    kps = [clamp01(v) if i % 3 != 2 else int(v) for i, v in enumerate(kps)]

    out_line = [str(dst_class), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for i in range(0, len(kps), 3):
        out_line.append(f"{kps[i]:.6f}")
        out_line.append(f"{kps[i+1]:.6f}")
        out_line.append(str(kps[i+2]))

    dst_image_dir = os.path.join(OUTPUT_IMAGES_DIR, split_name)
    dst_label_dir = os.path.join(OUTPUT_LABELS_DIR, split_name)

    dst_image_path = os.path.join(dst_image_dir, os.path.basename(image_path))
    dst_label_path = os.path.join(dst_label_dir, stem + ".txt")

    shutil.copy2(image_path, dst_image_path)
    with open(dst_label_path, "w", encoding="utf-8") as f:
        f.write(" ".join(out_line) + "\n")

    return True

def write_dataset_yaml():
    yaml_text = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val
names:
  0: id_card
kpt_shape: [4, 3]
flip_idx: [0, 1, 2, 3]
"""
    with open(os.path.join(OUTPUT_ROOT, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)

def main():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(os.path.join(OUTPUT_IMAGES_DIR, "train"))
    ensure_dir(os.path.join(OUTPUT_IMAGES_DIR, "val"))
    ensure_dir(os.path.join(OUTPUT_LABELS_DIR, "train"))
    ensure_dir(os.path.join(OUTPUT_LABELS_DIR, "val"))
    
    label_files = []
    for name in sorted(os.listdir(LABELS_DIR)):
        if Path(name).suffix.lower() == ".txt":
            label_files.append(name)

    random.seed(RANDOM_SEED)
    random.shuffle(label_files)

    split_idx = int(len(label_files) * TRAIN_RATIO)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    count_ok = 0
    count_total = 0

    for name in train_files:
        count_total += 1
        label_path = os.path.join(LABELS_DIR, name)
        if convert_one(label_path, "train"):
            count_ok += 1

    for name in val_files:
        count_total += 1
        label_path = os.path.join(LABELS_DIR, name)
        if convert_one(label_path, "val"):
            count_ok += 1

    write_dataset_yaml()
    print(f"converted {count_ok}/{count_total} files")
    print(f"train files: {len(train_files)}")
    print(f"val files: {len(val_files)}")
    print(f"output: {OUTPUT_ROOT}")
    print(f"yaml: {os.path.join(OUTPUT_ROOT, 'dataset.yaml')}")

if __name__ == "__main__":
    main()
