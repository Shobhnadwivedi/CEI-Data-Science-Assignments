"""
train.py
--------
Trains the denoising autoencoder and saves:
  - denoising_autoencoder.keras   (the trained model, used by app.py)
  - results.png                   (original vs noisy vs reconstructed grid)

Run:
    python train.py --epochs 20 --batch-size 128 --noise 0.5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import load_mnist, add_noise, build_autoencoder, NOISE_FACTOR, SEED


def plot_results(clean, noisy, denoised, n=10, out_path="results.png"):
    """Save a 3-row comparison grid: original / noisy / reconstructed."""
    plt.figure(figsize=(n * 1.5, 4.5))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(clean[i].squeeze(), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Original", loc="left", fontsize=10)

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].squeeze(), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Noisy", loc="left", fontsize=10)

        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i].squeeze(), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Denoised", loc="left", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved comparison grid to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MNIST denoising autoencoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=NOISE_FACTOR)
    parser.add_argument("--model-out", type=str, default="denoising_autoencoder.keras")
    args = parser.parse_args()

    tf.random.set_seed(SEED)

    # 1 & 2. Load + preprocess
    print("Loading MNIST...")
    x_train, x_test = load_mnist()

    # 3. Add artificial noise
    print(f"Adding Gaussian noise (factor={args.noise})...")
    x_train_noisy = add_noise(x_train, args.noise, seed=SEED)
    x_test_noisy = add_noise(x_test, args.noise, seed=SEED + 1)

    # 4. Build model
    autoencoder = build_autoencoder()
    autoencoder.summary()

    # 5. Train: noisy images as INPUT, clean images as TARGET
    print("Training...")
    autoencoder.fit(
        x_train_noisy, x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
    )

    # 6. Generate denoised outputs on the test set
    print("Reconstructing test images...")
    denoised = autoencoder.predict(x_test_noisy[:10])

    # Save artefacts
    autoencoder.save(args.model_out)
    print(f"Saved model to {args.model_out}")
    plot_results(x_test[:10], x_test_noisy[:10], denoised, n=10)


if __name__ == "__main__":
    main()
