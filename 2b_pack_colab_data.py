"""
Pack prepared YOLO data into data.zip for Google Colab training.

Run AFTER 1_download_dataset.py and 2_prepare_yolo.py.

The zip contains:
  images/train/  images/val/
  labels/train/  labels/val/
  dataset.yaml

Upload the result to Google Drive at: MyDrive/yolo_data/data.zip

Usage:
    python 2b_pack_colab_data.py
    python 2b_pack_colab_data.py --data-dir data --out data.zip
"""
import argparse
import zipfile
from pathlib import Path


def pack(data_dir: Path, yaml_path: Path, out_path: Path):
    folders = [
        data_dir / 'images' / 'train',
        data_dir / 'images' / 'val',
        data_dir / 'labels' / 'train',
        data_dir / 'labels' / 'val',
    ]
    for f in folders:
        if not f.exists():
            raise FileNotFoundError(
                f'{f} not found — run 1_download_dataset.py and 2_prepare_yolo.py first'
            )
    if not yaml_path.exists():
        raise FileNotFoundError(f'{yaml_path} not found')

    total = sum(1 for f in folders for _ in f.iterdir()) + 1
    print(f'Packing {total} files into {out_path} ...')

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            rel_root = folder.relative_to(data_dir)
            for file in sorted(folder.iterdir()):
                zf.write(file, rel_root / file.name)

        zf.write(yaml_path, 'dataset.yaml')

    size_mb = out_path.stat().st_size // 1024 // 1024
    print(f'Done: {out_path}  ({size_mb} MB)')
    print('Upload to Google Drive at: MyDrive/yolo_data/data.zip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data',
                        help='directory with images/ and labels/ (default: data)')
    parser.add_argument('--out', default='data.zip',
                        help='output zip path (default: data.zip)')
    args = parser.parse_args()

    root = Path(__file__).parent
    pack(
        data_dir=root / args.data_dir,
        yaml_path=root / 'dataset.yaml',
        out_path=root / args.out,
    )


if __name__ == '__main__':
    main()
