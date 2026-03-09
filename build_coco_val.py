import os
import json
import glob
from PIL import Image

def yolo_to_coco_keypoints(images_dir, labels_dir, output_json, class_names):
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": class_names[0], "keypoints": ["k1", "k2", "k3", "k4"], "skeleton": []}]
    }
    
    image_paths = glob.glob(os.path.join(images_dir, "*.*"))
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    ann_id = 0
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        with Image.open(img_path) as img:
            width, height = img.size
            
        img_dict_id = int(img_stem) if img_stem.isnumeric() else img_stem
            
        coco_dict["images"].append({
            "id": img_dict_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        label_path = os.path.join(labels_dir, img_stem + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = [float(x) for x in line.strip().split()]
            cls_id = int(parts[0])
            
            w = parts[3] * width
            h = parts[4] * height
            x = (parts[1] * width) - (w / 2)
            y = (parts[2] * height) - (h / 2)
            
            keypoints = []
            num_kpts = 0
            for i in range(5, len(parts), 3):
                kx = parts[i] * width
                ky = parts[i+1] * height
                kv = int(parts[i+2])
                keypoints.extend([kx, ky, kv])
                if kv > 0:
                    num_kpts += 1
                    
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_dict_id,
                "category_id": cls_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": num_kpts
            })
            ann_id += 1
            
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_dict, f)

if __name__ == "__main__":
    yolo_to_coco_keypoints(
        images_dir="../crop-dataset-eitan-yolo/images/val",
        labels_dir="../crop-dataset-eitan-yolo/labels/val",
        output_json="../crop-dataset-eitan-yolo/annotations/val.json",
        class_names=["id_card"]
    )
