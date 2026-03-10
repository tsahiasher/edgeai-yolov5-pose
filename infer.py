import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import torch
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression
from torchvision.ops import nms

IMAGE_PATH = "001.jpg"
WEIGHTS_PATH = "runs/train/exp/weights/best.pt"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
NC = 1
NKPT = 4

def nms_pose(pred, conf_thres=0.25, iou_thres=0.45):
    scores = pred[:, 4] * pred[:, 5]
    mask = scores > conf_thres

    pred = pred[mask]
    scores = scores[mask]

    if pred.shape[0] == 0:
        return pred

    boxes = pred[:, :4].clone()
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_thres)

    return pred[keep]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(WEIGHTS_PATH, weights_only=False, map_location=device)
model = ckpt["model"].float().eval().to(device)

img0 = cv2.imread(IMAGE_PATH)
h0, w0 = img0.shape[:2]

img = letterbox(img0, (IMG_SIZE, IMG_SIZE), stride=64, auto=False)[0]
img = img[:, :, ::-1].transpose(2, 0, 1).copy()
img = torch.from_numpy(img).float() / 255.0
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img)[0]

pred = non_max_suppression(
    pred,
    conf_thres=CONF_THRES,
    iou_thres=IOU_THRES,
    kpt_label=True,
    nc=NC,
    nkpt=NKPT
)

det = pred[0]

if det is None or len(det) == 0:
    print("no detection")
    exit()

gain = min(IMG_SIZE / h0, IMG_SIZE / w0)
pad_x = (IMG_SIZE - w0 * gain) / 2.0
pad_y = (IMG_SIZE - h0 * gain) / 2.0

boxes = det[:, :4].clone()
boxes[:, [0, 2]] -= pad_x
boxes[:, [1, 3]] -= pad_y
boxes[:, [0, 2]] /= gain
boxes[:, [1, 3]] /= gain
boxes[:, [0, 2]].clamp_(0, w0 - 1)
boxes[:, [1, 3]].clamp_(0, h0 - 1)

scores = det[:, 4].clone()
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

print("detections:")
for i in range(det.shape[0]):
    print(
        i,
        "score=", round(scores[i].item(), 4),
        "area=", round(areas[i].item(), 1),
        "box=", [round(v, 1) for v in boxes[i].tolist()]
    )

best_idx = torch.argmax(areas)
best_box = boxes[best_idx]
best_det = det[best_idx].clone()

best_kpts = best_det[6:18].reshape(4, 3)
best_kpts[:, 0] -= pad_x
best_kpts[:, 1] -= pad_y
best_kpts[:, 0] /= gain
best_kpts[:, 1] /= gain
best_kpts[:, 0].clamp_(0, w0 - 1)
best_kpts[:, 1].clamp_(0, h0 - 1)

print("chosen idx:", best_idx.item())
print("chosen score:", scores[best_idx].item())
print("chosen box:", [round(v, 1) for v in best_box.tolist()])
for i in range(4):
    print(f"kpt{i}: ({best_kpts[i,0]:.1f}, {best_kpts[i,1]:.1f}) conf={best_kpts[i,2]:.3f}")

vis = img0.copy()

pts = []
for i in range(4):
    x = int(best_kpts[i, 0].item())
    y = int(best_kpts[i, 1].item())
    pts.append((x, y))
    cv2.circle(vis, (x, y), 7, (0, 255, 0), -1)
    cv2.putText(vis, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
    cv2.line(vis, pts[a], pts[b], (255, 0, 0), 2)

out_path = "single_image_pose_result.jpg"
cv2.imwrite(out_path, vis)
print("saved:", out_path)