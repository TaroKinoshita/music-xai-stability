

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

import umap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_model.h5"
CSV_PATH   = ROOT / "results" / "csv" / "selected_50.csv"
SPEC_DIR   = ROOT / "results" / "Spectrogram"
STABILITY_CSV = ROOT / "results" / "stability" / "stability_scores_final.csv"
OUTPUT_DIR = ROOT / "results" / "umap"

CNN_HEIGHT = 128
CNN_WIDTH  = 677

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

flatten_layer = None
for layer in model.layers:
    if "flatten" in layer.name:
        flatten_layer = layer
        break

flatten_idx = model.layers.index(flatten_layer)
feature_extractor = Model(
    inputs=model.layers[0].input,
    outputs=model.layers[flatten_idx].output
)
dummy = np.zeros((1, CNN_HEIGHT, CNN_WIDTH, 1), dtype=np.float32)
dummy_out = feature_extractor.predict(dummy, verbose=0)


df = pd.read_csv(CSV_PATH)
stability_df = pd.read_csv(STABILITY_CSV)
stability_df = stability_df[stability_df["status"] == "success"]

df = df.merge(
    stability_df[["song_name", "LIME_CV", "IG_CV"]],
    left_on=df["filename"].str.replace(".npy", ""),
    right_on="song_name",
    how="left"
)

print(f"{len(df)} songs loaded")


features = []
valid_idx = []

for idx, row in df.iterrows():
    spec_path = os.path.join(SPEC_DIR, row["filename"])

    if not os.path.exists(spec_path):
        print(f"  Errror:  {row['filename']}")
        continue

    spec = np.load(spec_path)
    if spec.ndim == 3:
        spec = spec[:, :, 0]


    spec_4d = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    spec_resized = tf.image.resize(spec_4d, (CNN_HEIGHT, CNN_WIDTH)).numpy()

    feat = feature_extractor.predict(spec_resized, verbose=0)[0]
    features.append(feat)
    valid_idx.append(idx)

features = np.array(features)
df_valid = df.loc[valid_idx].reset_index(drop=True)



reducer = umap.UMAP(
    n_components=2,   
    n_neighbors=10,   
    min_dist=0.3,     
    random_state=42  
)

embedding = reducer.fit_transform(features)


genre_colors ={
    "classical": "#4C9BE8",  
    "hiphop":    "#E85C4C",  
    "jazz":      "#2DB37A",  
    "metal":     "#9B4CE8",  
    "rock":      "#E8A84C",   }

fig, ax = plt.subplots(figsize=(9, 7))

for genre, color in genre_colors.items():
    mask = df_valid["true_genre"] == genre
    ax.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=color, label=genre.capitalize(),
        s=80, alpha=0.8, edgecolors="white", linewidths=0.5
    )

boundary_mask = df_valid["type"] == "boundary"
ax.scatter(
    embedding[boundary_mask, 0], embedding[boundary_mask, 1],
    facecolors="none", edgecolors="black",
    s=140, linewidths=1.5, label="Boundary song", zorder=5
)

ax.set_title("Feature Space: 50 Songs (UMAP 2D)", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "umap_by_genre.png")
plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(9, 7))

type_colors = {"prototypical": "#4C9BE8", "boundary": "#E85C4C"}
type_markers = {"prototypical": "o", "boundary": "^"}

for song_type, color in type_colors.items():
    mask = df_valid["type"] == song_type
    ax.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=color, marker=type_markers[song_type],
        label=song_type.capitalize(),
        s=90, alpha=0.8, edgecolors="white", linewidths=0.5
    )

ax.set_title("Feature Space: Prototypical vs Boundary Songs", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "umap_by_type.png")
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(9, 7))

lime_cv = df_valid["LIME_CV"].fillna(df_valid["LIME_CV"].median()).values

sizes = 30 + lime_cv * 300

scatter = ax.scatter(
    embedding[:, 0], embedding[:, 1],
    c=lime_cv, cmap="RdYlGn_r",   
    s=sizes, alpha=0.8,
    edgecolors="white", linewidths=0.5
)

plt.colorbar(scatter, ax=ax, label="LIME CV (higher = more unstable)")


ax.scatter(
    embedding[boundary_mask, 0], embedding[boundary_mask, 1],
    facecolors="none", edgecolors="black",
    s=sizes[boundary_mask] + 40, linewidths=1.5,
    label="Boundary song", zorder=5 )

ax.set_title("Feature Space: Colored by LIME Instability", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "umap_by_stability.png")
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close()


print("\n Output :", OUTPUT_DIR)
