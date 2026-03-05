"""Gradio app for Hurricane Debris cascaded inference."""

import argparse
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from hurricane_debris.config import ExperimentConfig
from hurricane_debris.models.cascade import CascadedInference


def _load_pipeline(florence_dir: str, sam2_checkpoint: str, device: str):
    config = ExperimentConfig(device=device)
    return CascadedInference(
        florence_model_dir=florence_dir,
        sam2_checkpoint=sam2_checkpoint,
        config=config,
        device=config.resolve_device(),
    )


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio app for cascaded debris inference")
    parser.add_argument("--florence-dir", default="./models/florence2_debris")
    parser.add_argument("--sam2-checkpoint", default="./models/sam2_debris/best_model.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    try:
        pipeline = _load_pipeline(args.florence_dir, args.sam2_checkpoint, args.device)
        init_error = ""
    except Exception as exc:
        pipeline = None
        init_error = f"Pipeline failed to load: {exc}"

    def infer(image, query, score_threshold):
        if image is None:
            return "Please upload an image.", ""
        if pipeline is None:
            return "", init_error

        temp_path = Path("./outputs/gradio")
        temp_path.mkdir(parents=True, exist_ok=True)
        image_path = temp_path / "input.png"
        image.save(image_path)

        result = pipeline.run(
            str(image_path),
            query=query.strip() if query else None,
            score_threshold=float(score_threshold),
        )
        return json.dumps(result.to_json(), indent=2), json.dumps(result.to_geojson(), indent=2)

    with gr.Blocks(title="Hurricane Debris Detection Demo") as demo:
        gr.Markdown("# Hurricane Debris Detection Demo")
        gr.Markdown(
            "Florence-2 + SAM2 cascaded inference with JSON and image-pixel GeoJSON output."
        )
        if init_error:
            gr.Markdown(f"**Warning:** {init_error}")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input UAV Image")
            with gr.Column():
                query_input = gr.Textbox(
                    label="Open-vocabulary Query",
                    value="debris, damaged building, flooded area, downed tree, damaged road, vehicle wreckage",
                )
                score_input = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Score Threshold")
                run_btn = gr.Button("Run Inference")

        json_out = gr.Code(label="Structured JSON", language="json")
        geojson_out = gr.Code(label="GeoJSON (image_pixel)", language="json")

        run_btn.click(
            fn=infer,
            inputs=[image_input, query_input, score_input],
            outputs=[json_out, geojson_out],
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
