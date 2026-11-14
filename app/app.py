import argparse
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO


def build_app(weights_path: str):
    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. Train a model or provide a valid --weights path."
        )

    model = YOLO(str(weights))

    def detect(image, confidence):
        if image is None:
            return None, "Please upload an image."

        results = model.predict(image, conf=confidence, verbose=False)
        result = results[0]
        plotted = result.plot()[:, :, ::-1]
        output = Image.fromarray(plotted)

        names = result.names
        counts = {}
        for cls_id in result.boxes.cls.tolist() if result.boxes is not None else []:
            label = names[int(cls_id)]
            counts[label] = counts.get(label, 0) + 1

        summary = "No defects detected." if not counts else "\n".join(
            f"{label}: {count}" for label, count in sorted(counts.items())
        )
        return output, summary

    with gr.Blocks(title="YOLO Industrial Defect Detection") as demo:
        gr.Markdown("# YOLO Industrial Defect Detection")
        gr.Markdown("Upload an image to locate and classify surface defects.")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            output_image = gr.Image(type="pil", label="Detection result")
        confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
        summary = gr.Textbox(label="Detection summary", lines=5)
        run_button = gr.Button("Run detection")
        run_button.click(detect, inputs=[input_image, confidence], outputs=[output_image, summary])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_app(args.weights)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
