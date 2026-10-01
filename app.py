from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "efficientnet_best_model.keras"
MODEL_LOAD_PATH = BASE_DIR / "efficientnet_best_model"
LABELS_PATH = BASE_DIR / "class_names.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_CACHE = None
LABELS_CACHE = None


def get_model_load_path():
    if MODEL_LOAD_PATH.exists():
        return MODEL_LOAD_PATH

    if MODEL_PATH.exists() and MODEL_PATH.is_dir():
        copytree(MODEL_PATH, MODEL_LOAD_PATH, dirs_exist_ok=True)
        return MODEL_LOAD_PATH

    if MODEL_PATH.exists() and MODEL_PATH.is_file():
        return MODEL_PATH

    raise FileNotFoundError(f"Model not found in project root: {MODEL_PATH}")


def load_model_and_labels():
    global MODEL_CACHE, LABELS_CACHE

    if MODEL_CACHE is None:
        model_source = get_model_load_path()
        MODEL_CACHE = tf.keras.models.load_model(str(model_source), compile=False)

    if LABELS_CACHE is None:
        with LABELS_PATH.open("r", encoding="utf-8") as file:
            LABELS_CACHE = json.load(file)

    return MODEL_CACHE, LABELS_CACHE


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_name = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            uploaded_name = file.filename
            image_array = preprocess_image(file)
            model, labels = load_model_and_labels()
            probabilities = model.predict(image_array, verbose=0)[0]
            best_index = int(np.argmax(probabilities))
            prediction = {
                "label": labels[best_index],
                "confidence": round(float(probabilities[best_index]) * 100, 2),
            }

    return render_template("index.html", prediction=prediction, uploaded_name=uploaded_name)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
