"""
model.py
--------
Shared utilities for the MNIST Denoising Autoencoder project.

Keeping data loading, noise generation, and the model definition in one place
means the notebook, the training script, and the Streamlit app all use the exact
same logic (reproducibility + no copy/paste drift).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Default Gaussian noise strength used across the project.
NOISE_FACTOR = 0.5
SEED = 42


def load_mnist():
    """Load MNIST, scale pixels to [0, 1], and add a channel dimension.

    Returns
    -------
    (x_train, x_test) : tuple of np.ndarray
        Clean images with shape (N, 28, 28, 1), dtype float32, range [0, 1].
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Normalise to [0, 1] so a sigmoid output layer can reproduce them.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Conv layers expect a channel axis: (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test


def add_noise(images, noise_factor=NOISE_FACTOR, seed=None):
    """Add Gaussian noise to images and clip back into the valid [0, 1] range.

    Parameters
    ----------
    images : np.ndarray
        Clean images in [0, 1].
    noise_factor : float
        Standard deviation multiplier for the Gaussian noise.
    seed : int, optional
        Set for reproducible noise.

    Returns
    -------
    np.ndarray
        Noisy images, same shape, clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    noisy = images + noise_factor * rng.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")


def build_autoencoder():
    """Build a convolutional denoising autoencoder for 28x28 grayscale images.

    Encoder compresses the image into a compact feature map; the decoder
    reconstructs a clean image from it. Using convolutions (rather than dense
    layers) preserves spatial structure and gives much sharper reconstructions.
    """
    inputs = layers.Input(shape=(28, 28, 1), name="noisy_input")

    # ---------------- Encoder ----------------
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)          # 28x28 -> 14x14
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)    # 14x14 -> 7x7

    # ---------------- Decoder ----------------
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)                          # 7x7 -> 14x14
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)                          # 14x14 -> 28x28
    outputs = layers.Conv2D(1, (3, 3), activation="sigmoid",
                            padding="same", name="clean_output")(x)

    autoencoder = models.Model(inputs, outputs, name="denoising_autoencoder")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder
