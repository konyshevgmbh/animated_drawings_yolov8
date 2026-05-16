"""
Visualise YOLO pose labels on training images.

Usage:
    python training/visualize_labels.py                    # 6 random train images
    python training/visualize_labels.py --split val -n 4   # 4 random val images
    python training/visualize_labels.py --out preview.png  # save instead of show
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# COCO-17 skeleton pairs (0-indexed)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

KPT_NAMES = [
    'nose', 'l.eye', 'r.eye', 'l.ear', 'r.ear',
    'l.sho', 'r.sho', 'l.elb', 'r.elb', 'l.wri', 'r.wri',
    'l.hip', 'r.hip', 'l.kne', 'r.kne', 'l.ank', 'r.ank',
]

# One colour per keypoint (BGR for cv2)
KPT_COLORS_BGR = [
    (0, 0, 255), (0, 85, 255), (0, 170, 255), (0, 255, 255), (0, 255, 170),
    (0, 255, 85), (0, 255, 0), (85, 255, 0), (170, 255, 0), (255, 255, 0),
    (255, 170, 0), (255, 85, 0), (255, 0, 0), (255, 0, 85), (255, 0, 170),
    (170, 0, 255), (85, 0, 255),
]


def draw_pose(img_bgr: np.ndarray, label_path: Path) -> np.ndarray:
    """Draw all YOLO pose annotations from label_path onto a copy of img_bgr."""
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        vals = list(map(float, line.split()))
        if len(vals) < 5 + 17 * 3:
            continue

        # Bounding box
        cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Keypoints
        kpts = vals[5:]
        pts = []
        for k in range(17):
            kx, ky, kv = kpts[k * 3], kpts[k * 3 + 1], kpts[k * 3 + 2]
            px = int(kx * w); py = int(ky * h)
            pts.append((px, py, kv))

        # Skeleton lines
        for (j1, j2) in SKELETON:
            px1, py1, v1 = pts[j1]
            px2, py2, v2 = pts[j2]
            if v1 > 0 and v2 > 0:
                cv2.line(out, (px1, py1), (px2, py2), (80, 200, 80), 2)

        # Dots
        for k, (px, py, v) in enumerate(pts):
            if v > 0:
                color = KPT_COLORS_BGR[k]
                cv2.circle(out, (px, py), 6, color, -1)
                cv2.circle(out, (px, py), 6, (0, 0, 0), 1)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default='data', help='dataset root')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('-n',      type=int, default=6, help='number of images')
    parser.add_argument('--out',   default=None, help='save path (e.g. preview.png)')
    parser.add_argument('--seed',  type=int, default=None)
    args = parser.parse_args()

    data      = Path(args.data)
    img_dir   = data / 'images' / args.split
    label_dir = data / 'labels' / args.split

    imgs = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
    imgs = [p for p in imgs if (label_dir / (p.stem + '.txt')).exists()]
    if not imgs:
        sys.exit(f'No labelled images found in {img_dir}')

    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(imgs, min(args.n, len(imgs)))

    cols = min(3, len(sample))
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for i, img_path in enumerate(sample):
        label_path = label_dir / (img_path.stem + '.txt')
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            axes[i].text(0.5, 0.5, 'cannot read', ha='center', va='center')
            axes[i].axis('off')
            continue

        annotated = draw_pose(img_bgr, label_path)
        axes[i].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        n_anns = len(label_path.read_text().strip().splitlines())
        axes[i].set_title(f'{img_path.name}\n{n_anns} annotation(s)', fontsize=8)
        axes[i].axis('off')

    for j in range(len(sample), len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f'YOLO pose labels — {args.split}  ({len(sample)} of {len(imgs)})',
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
