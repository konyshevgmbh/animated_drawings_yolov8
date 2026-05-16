"""
Download Amateur Drawings Dataset from Meta.

Downloads full tar archive (~50 GB) to data/raw/, then extracts all images.
After extraction, splits images into train/val sets.

Usage:
    python 1_download_dataset.py [--train-count 3000] [--val-count 500]
    python 1_download_dataset.py --skip-download  # if tar already present
"""
import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

ANNOTATIONS_URL  = 'https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings_annotations.json'
TAR_URL          = 'https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings.tar'
DATA_DIR         = Path(__file__).parent / 'data'
ANNOTATIONS_FILE = DATA_DIR / 'amateur_drawings_annotations.json'
TAR_FILE         = DATA_DIR / 'raw' / 'amateur_drawings.tar'
IMAGES_RAW       = DATA_DIR / 'raw' / 'images'


def _progress(count, block, total):
    done = count * block
    pct = min(100, 100 * done // total) if total > 0 else 0
    mb_done  = done // 1024 // 1024
    mb_total = total // 1024 // 1024
    print(f'  {pct}%  {mb_done} / {mb_total} MB', flush=True)


def download_annotations():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ANNOTATIONS_FILE.exists():
        print(f'  Annotations already present ({ANNOTATIONS_FILE.stat().st_size // 1024 // 1024} MB) — skipping.')
        return
    print(f'  Downloading annotations...')
    urlretrieve(ANNOTATIONS_URL, ANNOTATIONS_FILE, _progress)
    print(f'  Saved: {ANNOTATIONS_FILE}')


def download_tar():
    TAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TAR_FILE.exists():
        print(f'  Tar already present ({TAR_FILE.stat().st_size // 1024 // 1024 // 1024} GB) — skipping download.')
        return
    print(f'  Downloading {TAR_URL}')
    print(f'  Saving to {TAR_FILE}  (~50 GB, progress every 1%)')
    urlretrieve(TAR_URL, TAR_FILE, _progress)
    print(f'  Saved: {TAR_FILE}')


def extract_tar():
    if IMAGES_RAW.exists() and any(IMAGES_RAW.rglob('*.png')):
        count = sum(1 for _ in IMAGES_RAW.rglob('*.png'))
        print(f'  Raw images already extracted ({count} files) — skipping.')
        return
    IMAGES_RAW.mkdir(parents=True, exist_ok=True)
    print(f'  Extracting {TAR_FILE} -> {IMAGES_RAW}')
    with tarfile.open(TAR_FILE, 'r') as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            tar.extract(member, IMAGES_RAW)
            if i % 5000 == 0:
                print(f'  {i}/{total} extracted...', flush=True)
    count = sum(1 for _ in IMAGES_RAW.rglob('*.png'))
    print(f'  Extraction done: {count} images')


def pick_ids(annotations: dict, train_n: int, val_n: int, seed: int = 42):
    annotated = {a['image_id'] for a in annotations['annotations'] if a.get('num_keypoints', 0) > 0}
    ids = sorted(annotated)
    random.seed(seed)
    random.shuffle(ids)
    total = min(train_n + val_n, len(ids))
    if total < train_n + val_n:
        print(f'  Only {len(ids)} usable images — adjusting.')
        train_n = int(total * 0.857)
        val_n   = total - train_n
    return ids[:train_n], ids[train_n:train_n + val_n]


def copy_split(id_to_img: dict, img_ids: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for img_id in img_ids:
        info = id_to_img.get(img_id)
        if not info:
            missing += 1
            continue
        src = IMAGES_RAW / info['file_name']   # e.g. raw/images/amateur_drawings/c/abc.png
        dst = out_dir / Path(info['file_name']).name
        if dst.exists():
            copied += 1
            continue
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    print(f'  {out_dir.name}: {copied} images, {missing} missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-count',    type=int,  default=3000)
    parser.add_argument('--val-count',      type=int,  default=500)
    parser.add_argument('--skip-download',  action='store_true',
                        help='skip tar download (if already on disk)')
    args = parser.parse_args()

    print('=== Step 1a: annotations ===')
    download_annotations()

    if not args.skip_download:
        print('\n=== Step 1b: tar archive (~50 GB) ===')
        download_tar()

    print('\n=== Step 1c: extract ===')
    extract_tar()

    print('\n=== Loading annotations ===')
    with open(ANNOTATIONS_FILE) as f:
        ann = json.load(f)
    print(f'  Images: {len(ann["images"])}   Annotations: {len(ann["annotations"])}')
    id_to_img = {img['id']: img for img in ann['images']}

    print('\n=== Selecting & copying train/val splits ===')
    train_ids, val_ids = pick_ids(ann, args.train_count, args.val_count)
    print(f'  Train: {len(train_ids)}   Val: {len(val_ids)}')
    copy_split(id_to_img, train_ids, DATA_DIR / 'images' / 'train')
    copy_split(id_to_img, val_ids,   DATA_DIR / 'images' / 'val')

    (DATA_DIR / 'splits.json').write_text(json.dumps({'train': train_ids, 'val': val_ids}))
    print('\nDone. Run 2_prepare_yolo.py next.')


if __name__ == '__main__':
    main()
