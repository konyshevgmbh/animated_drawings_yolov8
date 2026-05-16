"""
Fine-tune YOLOv8n-pose on Amateur Drawings Dataset.

Usage:
    python train.py [--epochs 50] [--batch 16] [--imgsz 640] [--device cpu]
    python train.py --epochs 1 --batch 4  # quick smoke-test

Output: training/runs/pose/drawn_humanoid_poseN/weights/best.pt
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch',  type=int, default=16)
    parser.add_argument('--imgsz',  type=int, default=640)
    parser.add_argument('--device', default='cpu',
                        help='cuda device id (e.g. 0) or "cpu"')
    parser.add_argument('--model',  default='yolov8n-pose.pt',
                        help='pretrained checkpoint to start from')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    from ultralytics import YOLO

    dataset_yaml = str(Path(__file__).parent / 'dataset.yaml')
    runs_dir     = str(Path(__file__).parent / 'runs')

    print(f'Model:   {args.model}')
    print(f'Dataset: {dataset_yaml}')
    print(f'Epochs:  {args.epochs}   Batch: {args.batch}   Imgsz: {args.imgsz}')
    print(f'Device:  {args.device}')
    print()

    model = YOLO(args.model)
    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=runs_dir,
        name='drawn_humanoid_pose',
        exist_ok=True,
        patience=15,         # early stopping
        save_period=10,      # save checkpoint every N epochs
        plots=True,
        verbose=True,
    )

    best = Path(runs_dir) / 'drawn_humanoid_pose' / 'weights' / 'best.pt'
    print(f'\nTraining complete. Best model: {best}')
    return results


if __name__ == '__main__':
    main()
