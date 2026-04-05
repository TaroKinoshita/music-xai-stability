

"""
Moderate noise (3% instead of 5%)
More trials (15 instead of 10) → more stable CV estimates
Consistent segmentation
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import slic

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_model.h5"
CSV_PATH   = ROOT / "results" / "csv" / "selected_50.csv"
SPEC_DIR   = ROOT / "results" / "Spectrogram"
OUTPUT_CSV = ROOT / "results" / "stability" / "stability_scores_final.csv"

CNN_HEIGHT = 128
CNN_WIDTH  = 677
 
noise_strength = 0.03  
num_trials = 15        
num_segments = 16
random_seed = 42       


print("STABILITY TEST start")
print("")
print(f"Noise strength: {noise_strength*100}%")
print(f"Trials per song: {num_trials}")
print("")
print()

model = load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} songs\n")


def predict_fn(images):
    gray = images[:, :, :, 0:1]
    resized = tf.image.resize(gray, (CNN_HEIGHT, CNN_WIDTH)).numpy()
    return model.predict(resized, verbose=0)

def compute_lime_cv_robust(lime_maps_list, segments_ref):
    """
    Remove outlier trials (top/bottom 10%) and Use only significant segments (top 8 out of 16)
    Weighted average by segment importance
    """
    segment_cvs = []
    segment_weights = []
    
    for seg_id in range(num_segments):
        seg_mask = (segments_ref == seg_id)
        
        scores = []
        for lime_map in lime_maps_list:
            seg_score = np.mean(lime_map[seg_mask])
            scores.append(seg_score)
        
        scores = np.array(scores)
        
 
        #n_keep = int(len(scores) * 0.8)
        #sorted_idx = np.argsort(np.abs(scores))
        #keep_idx = sorted_idx[:n_keep]
        #scores_filtered = scores[keep_idx]

        low = np.quantile(scores, 0.10)
        high = np.quantile(scores, 0.90)
        scores_filtered = scores[(scores >= low) & (scores <= high)]
        
        mean_score = np.mean(scores_filtered)
        std_score = np.std(scores_filtered)
        
        if abs(mean_score) > 1e-6:
            cv = std_score / abs(mean_score)
            segment_cvs.append(cv)
            segment_weights.append(abs(mean_score))  
        else:
            segment_cvs.append(0.0)
            segment_weights.append(0.0)
    
    segment_cvs = np.array(segment_cvs)
    segment_weights = np.array(segment_weights)
    
    if segment_weights.sum() > 0:

        top_8_idx = np.argsort(segment_weights)[-8:]
        weighted_cv = np.average(segment_cvs[top_8_idx], 
                                weights=segment_weights[top_8_idx])
    else:
        weighted_cv = np.mean(segment_cvs)
    
    return weighted_cv


def compute_ig_cv_robust(ig_maps_list):
    """
    Remove outlier trials and Use top 10% pixels 
    Spatial smoothing before CV calculation
    """

    array = np.array(ig_maps_list)
    
    n_keep = int(len(ig_maps_list) * 0.8)
    
    sorted_arrays = np.sort(array, axis=0)
    start_idx = (len(ig_maps_list) - n_keep) // 2
    end_idx = start_idx + n_keep
    array_filtered = sorted_arrays[start_idx:end_idx, :, :]

    mean = np.mean(array_filtered, axis=0)
    std = np.std(array_filtered, axis=0)
    
    from scipy.ndimage import gaussian_filter
    mean_smooth = gaussian_filter(mean, sigma=1.0)
    std_smooth = gaussian_filter(std, sigma=1.0)
    
    threshold = np.percentile(np.abs(mean_smooth).flatten(), 90)
    mask = np.abs(mean_smooth) >= threshold
    
    if mask.sum() > 10:
        cv_values = std_smooth[mask] / (np.abs(mean_smooth[mask]) + 1e-10)
        cv = np.mean(cv_values)
    else:
        cv_values = std_smooth / (np.abs(mean_smooth) + 1e-10)
        cv = np.mean(cv_values)
    
    return cv


results_list = []

np.random.seed(random_seed)

for song_idx, row in df.iterrows():
    
    filename = row["filename"]
    song_name = filename.replace(".npy", "")
    song_type = row["type"]
    
    print(f"\n{' '}")
    print(f"[{song_idx+1}/50] {song_name} ({song_type})")
    print(f"{' '}")
    
    spec_path = os.path.join(SPEC_DIR, filename)
    
    if not os.path.exists(spec_path):
        print("File not exists")
        continue
    
    spec = np.load(spec_path)
    
    if spec.ndim == 3:
        spec = spec[:, :, 0]
    

    spec_4d = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    spec_resized = tf.image.resize(spec_4d, (CNN_HEIGHT, CNN_WIDTH))
    spec = spec_resized[0, :, :, 0].numpy()
    
    try:
        lime_maps = []
        ig_maps = []
        segments_ref = None
        
        for trial_idx in range(num_trials):
            
            print(f"  Trial {trial_idx+1}/{num_trials}...", end=" ")
            
            noise = np.random.randn(CNN_HEIGHT, CNN_WIDTH)
            noise_scaled = noise * noise_strength * np.abs(spec).max()
            noisy_spec = spec + noise_scaled
            noisy_spec = np.maximum(noisy_spec, 0)
            
            spec_rgb = np.stack([noisy_spec] * 3, axis=-1)
            explainer = lime_image.LimeImageExplainer()
            
            explanation = explainer.explain_instance(
                image=spec_rgb,
                classifier_fn=predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000,
                segmentation_fn=lambda x: slic(x, n_segments=num_segments,
                                               compactness=0.1, sigma=1, 
                                               start_label=0)
            )
            
            top_label = explanation.top_labels[0]
            score_dict = dict(explanation.local_exp[top_label])
            segments = slic(spec_rgb, n_segments=num_segments,
                           compactness=0.1, sigma=1, start_label=0)
            
            if segments_ref is None:
                segments_ref = segments
            
            lime_map = np.zeros((CNN_HEIGHT, CNN_WIDTH), dtype=float)
            for seg_id in np.unique(segments):
                mask = (segments == seg_id)
                score = score_dict.get(seg_id, 0.0)
                lime_map[mask] = score
            
            lime_maps.append(lime_map)
            
            # IG
            input_tensor = tf.constant(noisy_spec[np.newaxis, :, :, np.newaxis],
                                      dtype=tf.float32)
            baseline = tf.zeros_like(input_tensor)
            
            pred = model(input_tensor, training=False).numpy()[0]
            top_label = int(np.argmax(pred))
            
            alphas = tf.linspace(0.0, 1.0, 51)
            interpolated = baseline + alphas[:, np.newaxis, np.newaxis, np.newaxis] * \
                          (input_tensor - baseline)
            
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                preds = model(interpolated, training=False)
                target_score = preds[:, top_label]
            
            grads = tape.gradient(target_score, interpolated)
            avg_grads = tf.reduce_mean(grads, axis=0)
            ig_map = (avg_grads * (input_tensor - baseline)).numpy()[0, :, :, 0]
            
            ig_maps.append(ig_map)
            
            print("DOne")
        
        print("\n  Computing CV")
        
        lime_cv = compute_lime_cv_robust(lime_maps, segments_ref)
        ig_cv = compute_ig_cv_robust(ig_maps)
        
        print(f"  LIME CV: {lime_cv:.4f}")
        print(f"  IG CV:   {ig_cv:.4f}")
        
        results_list.append({
            "song_name": song_name,
            "type": song_type,
            "LIME_CV": lime_cv,
            "IG_CV": ig_cv,
            "status": "success"
        })
        
    except Exception as e:
        print(f"\n Error: {e}")
        results_list.append({
            "song_name": song_name,
            "type": song_type,
            "LIME_CV": np.nan,
            "IG_CV": np.nan,
            "status": f"error: {e}"
        })


results_df = pd.DataFrame(results_list)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\n")
print("TEST done")
print(" ")
print(f"Saved to: {OUTPUT_CSV}")


proto = results_df[results_df["type"] == "prototypical"]
bound = results_df[results_df["type"] == "boundary"]

print("\n")
print("RESULTs")
print(" ")

print(f"\nPrototypical songs (n={len(proto)}):")
print(f"  LIME CV: {proto['LIME_CV'].mean():.4f} ± {proto['LIME_CV'].std():.4f}")
print(f"  IG CV:   {proto['IG_CV'].mean():.4f} ± {proto['IG_CV'].std():.4f}")

print(f"\nBoundary songs (n={len(bound)}):")
print(f"  LIME CV: {bound['LIME_CV'].mean():.4f} ± {bound['LIME_CV'].std():.4f}")
print(f"  IG CV:   {bound['IG_CV'].mean():.4f} ± {bound['IG_CV'].std():.4f}")

from scipy import stats
t_lime, p_lime = stats.ttest_ind(proto['LIME_CV'], bound['LIME_CV'])
t_ig, p_ig = stats.ttest_ind(proto['IG_CV'], bound['IG_CV'])

print("\n" )
print("STATISTICAL test")
print(" ")
print(f"\n  LIME (Prototypical vs Boundary):")
print(f"  t = {t_lime:.4f}")
print(f"  p = {p_lime:.6f}")
if p_lime < 0.05:
    print(f" SIGNIFICANT (p < 0.05)")
else:
    print(f" Not significant (p >= 0.05)")

print(f"\nIG (Prototypical vs Boundary):")
print(f"  t = {t_ig:.4f}")
print(f"  p = {p_ig:.6f}")
if p_ig < 0.05:
    print(f" SIGNIFICANT (p < 0.05)")
else:
    print(f"Not significant (p >= 0.05)")

lime_cohens_d = (bound['LIME_CV'].mean() - proto['LIME_CV'].mean()) / \
                np.sqrt((proto['LIME_CV'].std()**2 + bound['LIME_CV'].std()**2) / 2)
ig_cohens_d = (bound['IG_CV'].mean() - proto['IG_CV'].mean()) / \
              np.sqrt((proto['IG_CV'].std()**2 + bound['IG_CV'].std()**2) / 2)

print("\n" )
print("EFFECT SIZE (Cohen's d)")
print(" ")
print(f"  LIME: d = {lime_cohens_d:.3f}")
print(f"  IG:   d = {ig_cohens_d:.3f}")
print("\n  Interpretation:")
print("  0.2 = small, 0.5 = medium, 0.8 = large")

if p_lime < 0.05 or p_ig < 0.05:
    print("\n" )
    print(" HYPOTHESIS supported")
    print(" ")

else:
    print("\n" )
    print("HYPOTHESIS not fullly supported ")
    print(" " )


print("\nFirst 10 songs:")
print(results_df[["song_name", "type", "LIME_CV", "IG_CV"]].head(10))