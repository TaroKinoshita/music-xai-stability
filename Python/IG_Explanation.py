

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_model.h5"
CSV_PATH   = Path(__file__).resolve().parent.parent / "results" / "csv" / "selected_50.csv"
SPEC_DIR   = Path(__file__).resolve().parent.parent / "results" / "Spectrogram"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "IG"

GENRE_LABELS = ["classical", "hiphop", "jazz", "metal", "rock"]

IG_STEPS         = 50   
CNN_INPUT_HEIGHT = 128
CNN_INPUT_WIDTH  = 677

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

df = pd.read_csv(CSV_PATH)

def resize_spec(spec_2d):
    """
    Resize the spectrogram from (128, 646) to (128, 677) to match the CNN input size
    """
    spec_4d = spec_2d[np.newaxis, :, :, np.newaxis].astype(np.float32)
    resized = tf.image.resize(spec_4d, (CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH))
    return resized[0, :, :, 0].numpy()  


def compute_integrated_gradients(spec_array):
    """
    Compute Integrated Gradients and use (original audio - silence) to get the importance

    """


    #change form (1, 128, 677, 1) for CNN
    input_tensor = tf.constant(
        spec_array[np.newaxis, :, :, np.newaxis],
        dtype=tf.float32)

    baseline = tf.zeros_like(input_tensor)
    pred = model(input_tensor, training=False).numpy()[0]

    top_label = int(np.argmax(pred))
    alphas = tf.linspace(0.0, 1.0, IG_STEPS + 1)

    interpolated = baseline + alphas[:, np.newaxis, np.newaxis, np.newaxis] \
                   * (input_tensor - baseline)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)

        preds = model(interpolated, training=False)

        target_preds = preds[:, top_label]

    gradients = tape.gradient(target_preds, interpolated)

    avg_gradients = tf.reduce_mean(gradients, axis=0)
    ig_map = (avg_gradients * (input_tensor - baseline)).numpy()
    ig_map = ig_map[0, :, :, 0]

    return ig_map, top_label

def save_ig_results(ig_map, spec_array, song_name, top_label):
    """
    Save the IG result as .npy and .png
    """
    base_path = os.path.join(OUTPUT_DIR, song_name)

    np.save(f"{base_path}_ig.npy", ig_map)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(spec_array, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"Original Spectrogram\n({song_name})")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Frequency Bin")

    vmax = np.abs(ig_map).max()
    im = axes[1].imshow(
        ig_map, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title(f"IG Importance Map\nPredicted: {GENRE_LABELS[top_label]}")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Frequency Bin")

    plt.tight_layout()
    plt.savefig(f"{base_path}_ig.png", dpi=100, bbox_inches="tight")
    plt.close()

    print(f" SAVE : {song_name}_ig.npy / .png")



print("IG Explanation Starts")

results_log = []

for idx, row in df.iterrows():

    filename   = row["filename"]
    song_type  = row["type"]
    genre      = row["true_genre"]
    confidence = row["confidence"]
    song_name  = filename.replace(".npy", "")

    print(f"\n[{idx+1}/50] {song_name} ({song_type}, {genre}, conf={confidence:.3f})")

    spec_path = os.path.join(SPEC_DIR, filename)
    if not os.path.exists(spec_path):
        print(f" file does not exists : {spec_path} ")
        continue

    spec_array = np.load(spec_path)
    if spec_array.ndim == 3:
        spec_array = spec_array[:, :, 0]   

    try:
        spec_resized = resize_spec(spec_array)

        ig_map, top_label = compute_integrated_gradients(spec_resized)

        save_ig_results(ig_map, spec_array, song_name, top_label)

        results_log.append({
            "filename":  filename,
            "song_name": song_name,
            "type":      song_type,
            "genre":     genre,
            "confidence":confidence,
            "top_label": GENRE_LABELS[top_label],
            "status":    "success"
        })

    except Exception as e:
        print(f" error: {e}")
        results_log.append({
            "filename":  filename,
            "song_name": song_name,
            "type":      song_type,
            "genre":     genre,
            "confidence":confidence,
            "top_label": "error",
            "status":    f"error: {e}"
        })


log_df = pd.DataFrame(results_log)
log_path = os.path.join(OUTPUT_DIR, "ig_generation_log.csv")
log_df.to_csv(log_path, index=False)

success = len(log_df[log_df["status"] == "success"])
fail    = len(log_df[log_df["status"] != "success"])

print("\n")
print("IG Explanation done")
