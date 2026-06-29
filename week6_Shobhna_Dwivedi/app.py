"""
app.py
------
Streamlit app for the MNIST Denoising Autoencoder.

Pick a test digit (or upload your own), add adjustable noise, and watch the
trained autoencoder clean it up — shown side by side.

Run:
    streamlit run app.py

Requires a trained model file `denoising_autoencoder.keras` in the same folder.
Create it by running `python train.py` or the notebook first.
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from model import add_noise

MODEL_PATH = "denoising_autoencoder.keras"

st.set_page_config(page_title="MNIST Denoising Autoencoder", page_icon="🧹", layout="wide")


@st.cache_resource
def get_model():
    """Load the trained model once and cache it."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def get_test_data():
    """Load and normalise the MNIST test set for the demo."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    return x_test, y_test


def preprocess_upload(uploaded_file):
    """Convert an uploaded image into a 28x28 grayscale [0,1] array."""
    img = Image.open(uploaded_file).convert("L")        # grayscale
    img = ImageOps.fit(img, (28, 28))                   # resize/crop to 28x28
    arr = np.array(img).astype("float32") / 255.0
    return arr


def denoise(model, image_2d, noise_factor):
    """Add noise to a single 28x28 image and run it through the model."""
    clean = image_2d.reshape(1, 28, 28, 1)
    noisy = add_noise(clean, noise_factor, seed=0)
    denoised = model.predict(noisy, verbose=0)
    return clean.squeeze(), noisy.squeeze(), denoised.squeeze()


# ----------------------------- UI -----------------------------
st.title("🧹 MNIST Denoising Autoencoder")
st.write(
    "A convolutional autoencoder trained to strip Gaussian noise off handwritten "
    "digits while keeping the digit shape intact."
)

model = get_model()
if model is None:
    st.error(
        f"Model file `{MODEL_PATH}` not found. "
        "Train it first with `python train.py` (or run the notebook), then reload."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    source = st.radio("Image source", ["MNIST test set", "Upload your own"])
    noise_factor = st.slider("Noise level", 0.0, 1.0, 0.5, 0.05)

    image_2d = None
    if source == "MNIST test set":
        x_test, y_test = get_test_data()
        idx = st.number_input(
            "Test image index", min_value=0, max_value=len(x_test) - 1, value=0, step=1
        )
        image_2d = x_test[idx]
        st.caption(f"True label: **{y_test[idx]}**")
    else:
        up = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
        if up is not None:
            image_2d = preprocess_upload(up)

if image_2d is None:
    st.info("Upload an image to get started.")
    st.stop()

clean, noisy, denoised = denoise(model, image_2d, noise_factor)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(clean, width=220, clamp=True)
with col2:
    st.subheader("Noisy input")
    st.image(noisy, width=220, clamp=True)
with col3:
    st.subheader("Denoised output")
    st.image(denoised, width=220, clamp=True)

# Simple reconstruction-quality readout
mse = float(np.mean((clean - denoised) ** 2))
psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
st.metric("Reconstruction PSNR (vs original)", f"{psnr:.2f} dB")
st.caption("Higher PSNR = closer to the clean original. Try moving the noise slider.")
