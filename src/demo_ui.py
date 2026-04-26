"""Interactive Gradio demo for GTSRB traffic sign classification.

Allows selecting between all five trained model variants, uploading an image,
and viewing the top-5 predictions with confidence scores.

Usage
-----
    # Install dependency once:
    pip install gradio

    # Run from the project root:
    python src/demo_ui.py
    python src/demo_ui.py --share   # creates a public URL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from model import BaselineCNN
from model_improved import DeepCNN, LeakyReLUCNN, MobileNetTransfer, StrideCNN

# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

SIGN_LABELS = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing (vehicles > 3.5t)", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No vehicles (> 3.5t)", "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all restrictions", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing (> 3.5t)",
]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_INFO = {
    "Baseline CNN": {
        "path": "models/baseline.pth",
        "class": BaselineCNN,
        "kwargs": {"num_classes": 43, "input_size": 32},
        "description": (
            "**Baseline CNN** — 3 convolutional blocks with BatchNorm, ReLU, and MaxPooling, "
            "followed by a fully connected classifier with Dropout(0.5). "
            "629K parameters. The starting point for all comparisons."
        ),
    },
    "Deep CNN ⭐ Best": {
        "path": "models/deep_cnn.pth",
        "class": DeepCNN,
        "kwargs": {"num_classes": 43},
        "description": (
            "**Deep CNN** — Adds a 4th convolutional block (256 filters) and a larger "
            "classifier (512 units). 936K parameters. Achieves the highest test accuracy "
            "of **99.81%** — only 11 wrong predictions out of 5,881 test images."
        ),
    },
    "LeakyReLU CNN": {
        "path": "models/leakyrelu_cnn.pth",
        "class": LeakyReLUCNN,
        "kwargs": {"num_classes": 43},
        "description": (
            "**LeakyReLU CNN** — Same architecture as the baseline but replaces all ReLU "
            "activations with Leaky ReLU (slope=0.01). Designed to prevent dead neurons "
            "where gradient becomes exactly zero. 629K parameters."
        ),
    },
    "Stride CNN": {
        "path": "models/stride_cnn.pth",
        "class": StrideCNN,
        "kwargs": {"num_classes": 43},
        "description": (
            "**Stride CNN** — Replaces fixed MaxPooling with learned strided convolutions "
            "(stride=2). The network learns how to downsample rather than using a fixed rule. "
            "823K parameters. Fastest model to train."
        ),
    },
    "MobileNetV2 (Transfer)": {
        "path": "models/mobilenetv2.pth",
        "class": MobileNetTransfer,
        "kwargs": {"num_classes": 43},
        "description": (
            "**MobileNetV2** — Pretrained on ImageNet (1.2M images, 1000 classes), "
            "fine-tuned on GTSRB with a custom classifier head. 2.56M parameters. "
            "Transfer learning from a general-purpose vision backbone."
        ),
    },
}

# ---------------------------------------------------------------------------
# Preprocessing (same as val/test transforms in training)
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.3337, 0.3064, 0.3171),
        std=(0.2672, 0.2564, 0.2629),
    ),
])

# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------

_model_cache: dict[str, torch.nn.Module] = {}


def load_model(model_name: str) -> torch.nn.Module:
    if model_name in _model_cache:
        return _model_cache[model_name]

    info = MODEL_INFO[model_name]
    model_path = Path(info["path"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure you have trained all models first (python src/train_improved.py)."
        )

    model = info["class"](**info["kwargs"])
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(image: Image.Image, model_name: str) -> tuple[str, dict, str]:
    """Run inference and return top-5 results.

    Returns:
        (top_prediction, confidences_dict, model_description)
    """
    if image is None:
        return "No image provided.", {}, MODEL_INFO[model_name]["description"]

    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        return str(e), {}, MODEL_INFO[model_name]["description"]

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)  # [1, 3, 32, 32]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top5_probs, top5_idx = probs.topk(5)
    top5 = {
        f"{SIGN_LABELS[int(idx)]}": float(prob)
        for prob, idx in zip(top5_probs, top5_idx)
    }

    best_class = SIGN_LABELS[int(top5_idx[0])]
    confidence = float(top5_probs[0]) * 100
    top_label = f"🚦 {best_class}  ({confidence:.1f}% confidence)"

    return top_label, top5, MODEL_INFO[model_name]["description"]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_interface():
    try:
        import gradio as gr
    except Exception as e:
        print(f"Could not import gradio: {e}")
        print("Try: pip install 'gradio==3.50.2'")
        sys.exit(1)

    with gr.Blocks(title="GTSRB Traffic Sign Classifier", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
            # 🚦 GTSRB Traffic Sign Classifier
            **CNN-based traffic sign recognition — KLU Machine Learning Project**

            Upload a traffic sign image and select a model to classify it.
            All models were trained on the German Traffic Sign Recognition Benchmark (GTSRB).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Traffic Sign Image",
                    height=250,
                )
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_INFO.keys()),
                    value="Deep CNN ⭐ Best",
                    label="Select Model",
                )
                predict_btn = gr.Button("🔍 Classify", variant="primary")

            with gr.Column(scale=1):
                prediction_output = gr.Label(
                    label="Top Prediction",
                    num_top_classes=1,
                )
                top5_output = gr.Label(
                    label="Top-5 Predictions",
                    num_top_classes=5,
                )

        model_description = gr.Markdown(
            value=MODEL_INFO["Deep CNN ⭐ Best"]["description"],
            label="Model Description",
        )

        # Update description when model changes
        model_dropdown.change(
            fn=lambda name: MODEL_INFO[name]["description"],
            inputs=model_dropdown,
            outputs=model_description,
        )

        # Run prediction
        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[prediction_output, top5_output, model_description],
        )

        # Also predict on image upload
        image_input.change(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=[prediction_output, top5_output, model_description],
        )

        gr.Markdown(
            """
            ---
            ### Model Comparison

            | Model | Test Accuracy | Parameters | Notes |
            |-------|:---:|:---:|-------|
            | Baseline CNN | 99.49% | 629K | Starting point |
            | **Deep CNN** ⭐ | **99.81%** | 936K | Best accuracy |
            | LeakyReLU CNN | 99.46% | 629K | Dead neuron fix |
            | Stride CNN | 99.52% | 823K | Fastest training |
            | MobileNetV2 | 99.66% | 2.56M | Transfer learning |
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GTSRB Gradio Demo UI")
    parser.add_argument("--share", action="store_true",
                        help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on (default: 7860)")
    args = parser.parse_args()

    demo = build_interface()
    print("\n" + "=" * 55)
    print("  GTSRB Traffic Sign Classifier — Demo UI")
    print("=" * 55)
    print(f"  Open in browser: http://localhost:{args.port}")
    if args.share:
        print("  Generating public URL …")
    print()

    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
