"""
Convert Amateur Drawings Dataset (COCO JSON) to YOLOv8-pose label format.

YOLO pose label per line:
  class cx cy w h  kp1x kp1y v1  kp2x kp2y v2  ...  kp17x kp17y v17
  (all coordinates normalized 0-1, visibility: 0=absent, 1=occluded, 2=visible)

Usage:
    python 2_prepare_yolo.py [--data-dir data]

NOTE: info['width']/info['height'] in Amateur Drawings = original photo dimensions,
NOT the distributed thumbnail size. Always read actual dimensions from the image file.
"""
import argparse
import json
import cv2
from pathlib import Path


ANNOTATIONS_FILE = Path(__file__).parent / 'data' / 'amateur_drawings_annotations.json'


def coco_to_yolo(ann: dict, actual_w: int, actual_h: int):
    """
    Convert a single COCO annotation to a YOLO pose label line.
    actual_w/actual_h must be the stored image file dimensions (not annotation metadata).
    Returns None if the annotation has no usable keypoints or bbox is outside image.
    """
    bbox = ann.get('bbox')         # [x, y, w, h] — absolute pixels in thumbnail space
    kpts = ann.get('keypoints')    # flat list: [x1,y1,v1, x2,y2,v2, ...]
    num_kpts = ann.get('num_keypoints', 0)

    if not bbox or not kpts or num_kpts == 0:
        return None

    bx, by, bw, bh = bbox
    cx = (bx + bw / 2) / actual_w
    cy = (by + bh / 2) / actual_h
    nw = bw / actual_w
    nh = bh / actual_h

    if not (0 < cx < 1 and 0 < cy < 1 and 0 < nw <= 1 and 0 < nh <= 1):
        return None  # bbox outside image — skip

    kpt_parts = []
    for i in range(0, len(kpts), 3):
        kx, ky, v = kpts[i], kpts[i + 1], kpts[i + 2]
        kpt_parts.append(f'{kx / actual_w:.6f} {ky / actual_h:.6f} {int(v)}')

    return f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} ' + ' '.join(kpt_parts)


def prepare(data_dir: Path):
    print('Loading annotations...')
    with open(ANNOTATIONS_FILE) as f:
        ann = json.load(f)

    id_to_img = {img['id']: img for img in ann['images']}

    # group annotations by image_id
    img_to_anns: dict[int, list] = {}
    for a in ann['annotations']:
        img_to_anns.setdefault(a['image_id'], []).append(a)

    splits_file = data_dir / 'splits.json'
    with open(splits_file) as f:
        splits = json.load(f)

    stats = {}
    for split, img_ids in splits.items():
        label_dir = data_dir / 'labels' / split
        label_dir.mkdir(parents=True, exist_ok=True)
        img_dir = data_dir / 'images' / split

        written = skipped = 0
        for img_id in img_ids:
            info = id_to_img.get(img_id)
            if info is None:
                continue

            filename = Path(info['file_name']).name  # flat name: abc123.png
            img_path = img_dir / filename

            # check image actually downloaded
            if not img_path.exists():
                skipped += 1
                continue

            # Read actual stored dimensions — annotation metadata records original photo
            # size which differs from the distributed thumbnail size.
            actual = cv2.imread(str(img_path))
            if actual is None:
                skipped += 1
                continue
            actual_h, actual_w = actual.shape[:2]

            anns_for_img = img_to_anns.get(img_id, [])
            lines = [coco_to_yolo(a, actual_w, actual_h) for a in anns_for_img]
            lines = [l for l in lines if l]

            if not lines:
                skipped += 1
                continue

            label_file = label_dir / (Path(filename).stem + '.txt')
            label_file.write_text('\n'.join(lines))
            written += 1

        stats[split] = {'written': written, 'skipped': skipped}
        print(f'  {split}: {written} labels written, {skipped} skipped')

    print('Done.')
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(Path(__file__).parent / 'data'))
    args = parser.parse_args()
    prepare(Path(args.data_dir))
