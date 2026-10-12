import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "brain_tumor_mobilenetv2.keras"
CLASS_PATH = APP_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="Brain MRI Image Classifier", page_icon="🧠", layout="centered")
st.title("Brain Tumor MRI Image Classification")
st.warning(
    "Educational prototype only. This model is not clinically validated and must not be used "
    "to diagnose, rule out, or guide treatment for any medical condition."
)

@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class mapping file not found: {CLASS_PATH.name}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    if not isinstance(classes, list) or len(classes) != int(model.output_shape[-1]):
        raise ValueError("class_names.json does not match the model output dimensions.")
    return model, classes

try:
    model, class_names = load_model_and_classes()
except Exception as exc:
    st.error(f"Could not load the model artifacts: {exc}")
    st.info("Place the trained .keras model and class_names.json beside app.py.")
    st.stop()

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is not None:
    try:
        with Image.open(uploaded) as image:
            image = image.convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            resized = image.resize(IMAGE_SIZE)
            array = np.asarray(resized, dtype=np.float32)
        # Model was trained with an internal Rescaling(1/127.5, offset=-1) layer.
        # Therefore provide raw RGB pixel values in the 0–255 range.
        batch = np.expand_dims(array, axis=0)
        probabilities = model.predict(batch, verbose=0)[0]
        if len(probabilities) != len(class_names):
            raise ValueError("Prediction output and class mapping have different lengths.")
        ranked = np.argsort(probabilities)[::-1]
        st.subheader("Model output")
        st.metric("Top predicted category", class_names[int(ranked[0])])
        st.caption(
            f"Model score: {float(probabilities[ranked[0]]):.1%}. "
            "This is not a clinically calibrated probability."
        )
        st.write("Scores for all categories")
        st.bar_chart(
            {class_names[int(i)]: float(probabilities[i]) for i in ranked}
        )
        st.caption("Automated output only; seek assessment from a qualified medical professional.")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process this image: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
