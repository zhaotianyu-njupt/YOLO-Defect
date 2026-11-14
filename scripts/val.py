import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate a trained YOLO detector.")
    parser.add_argument("--weights", required=True, help="Path to best.pt or another checkpoint")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
    )
    print(metrics)


if __name__ == "__main__":
    main()
