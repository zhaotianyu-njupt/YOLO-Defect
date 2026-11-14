import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO defect detector.")
    parser.add_argument("--data", default="configs/data.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained weights or model YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="e.g. 0, cpu; empty lets Ultralytics choose")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
