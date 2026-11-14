import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on images or videos.")
    parser.add_argument("--weights", required=True, help="Path to trained weights")
    parser.add_argument("--source", required=True, help="Image, video, folder, or stream source")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device or None,
        save=True,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
