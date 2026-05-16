"""
Run YOLO pose inference on a folder of images and display results.

Usage:
    python predict_visualize.py --images data/images/val
    python predict_visualize.py --images data/images/val -n 9 --conf 0.25
    python predict_visualize.py --images data/images/val --out preview.png
    python predict_visualize.py --model drawn_humanoid_pose.onnx --images data/images/val
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KPT_COLORS_BGR = [
    (0, 0, 255), (0, 85, 255), (0, 170, 255), (0, 255, 255), (0, 255, 170),
    (0, 255, 85), (0, 255, 0), (85, 255, 0), (170, 255, 0), (255, 255, 0),
    (255, 170, 0), (255, 85, 0), (255, 0, 0), (255, 0, 85), (255, 0, 170),
    (170, 0, 255), (85, 0, 255),
]


def draw_predictions(img_bgr: np.ndarray, result) -> tuple[np.ndarray, str]:
    vis = img_bgr.copy()
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return vis, 'no detection'

    conf_scores = (
        result.boxes.conf.cpu().numpy()
        if result.boxes is not None and len(result.boxes.conf)
        else []
    )

    for person_idx in range(len(result.keypoints.xy)):
        kpts = result.keypoints.xy[person_idx].cpu().numpy()  # (17, 2)
        conf_xy = (
            result.keypoints.conf[person_idx].cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(17)
        )

        pts = [(int(kpts[k][0]), int(kpts[k][1])) for k in range(17)]

        if result.boxes is not None and person_idx < len(result.boxes.xyxy):
            x1, y1, x2, y2 = result.boxes.xyxy[person_idx].cpu().numpy().astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

        for j1, j2 in SKELETON:
            if conf_xy[j1] > 0 and conf_xy[j2] > 0 and pts[j1] != (0, 0) and pts[j2] != (0, 0):
                cv2.line(vis, pts[j1], pts[j2], (80, 200, 80), 2)

        for k, (px, py) in enumerate(pts):
            if conf_xy[k] > 0 and (px, py) != (0, 0):
                color = KPT_COLORS_BGR[k]
                cv2.circle(vis, (px, py), 6, color, -1)
                cv2.circle(vis, (px, py), 6, (0, 0, 0), 1)

    n = len(result.keypoints.xy)
    top_conf = float(conf_scores[0]) if len(conf_scores) else 0.0
    label = f'{n} person(s)  conf={top_conf:.2f}'
    return vis, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='drawn_humanoid_pose.pt', help='path to model .pt or .onnx')
    parser.add_argument('--images', required=True, help='folder with images')
    parser.add_argument('-n',       type=int, default=6, help='number of images to show')
    parser.add_argument('--conf',   type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--seed',   type=int, default=None)
    parser.add_argument('--out',    default=None, help='save path instead of show')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('ultralytics not installed — run: pip install ultralytics')

    img_dir = Path(args.images)
    imgs = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
    if not imgs:
        sys.exit(f'No images found in {img_dir}')

    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(imgs, min(args.n, len(imgs)))

    print(f'Loading model: {args.model}')
    model = YOLO(args.model)

    cols = min(3, len(sample))
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for i, img_path in enumerate(sample):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            axes[i].text(0.5, 0.5, 'cannot read', ha='center', va='center')
            axes[i].axis('off')
            continue

        results = model(img_bgr, conf=args.conf, verbose=False)
        vis, subtitle = draw_predictions(img_bgr, results[0])

        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'{img_path.name}\n{subtitle}', fontsize=8)
        axes[i].axis('off')

    for j in range(len(sample), len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f'YOLO pose predictions — {Path(args.model).name}  ({len(sample)} images)',
        fontsize=13,
    )
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=120, bbox_inches='tight')
        print(f'Saved: {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
