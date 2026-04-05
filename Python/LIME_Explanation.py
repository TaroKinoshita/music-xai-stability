
import numpy as np                          
import pandas as pd                         
import matplotlib.pyplot as plt           
import os                                  

from tensorflow.keras.models import load_model   

from lime import lime_image                 
from skimage.segmentation import slic      


from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_model.h5"
CSV_PATH   = ROOT / "results" / "csv" / "selected_50.csv"
SPEC_DIR   = ROOT / "results" / "Spectrogram"
OUTPUT_DIR = ROOT / "results" / "LIME"


GENRE_LABELS = ["classical", "hiphop", "jazz", "metal", "rock"]


NUM_SAMPLES  = 1000   
NUM_SEGMENTS = 16    

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

df = pd.read_csv(CSV_PATH)

print(df.head())   

CNN_INPUT_HEIGHT = 128
CNN_INPUT_WIDTH  = 677

def predict_fn(images):
    """
    Prediction function for LIME
    """
    import tensorflow as tf
    gray_images = images[:, :, :, 0:1]
    resized = tf.image.resize(
        gray_images,
        size=(CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH)).numpy()

    predictions = model.predict(resized, verbose=0)

    return predictions 


def generate_lime_for_one_song(spec_array, song_name):
    """
    Generate a LIME explanation for one song

    """
    

    spec_rgb = np.stack([spec_array] * 3, axis=-1)

    
    explainer = lime_image.LimeImageExplainer()

    segments = slic(
        spec_rgb,
        n_segments=NUM_SEGMENTS,    
        compactness=0.1,           
        sigma=1,                    
        start_label=0     )
    
    explanation = explainer.explain_instance(
        image=spec_rgb,                  
        classifier_fn=predict_fn,       
        top_labels=1,                    
        hide_color=0,                    
        num_samples=NUM_SAMPLES,        
        segmentation_fn=lambda x: slic(  
            x, n_segments=NUM_SEGMENTS,
            compactness=0.1, sigma=1, start_label=0
        )
    )
    

    top_label = explanation.top_labels[0]

    _, importance_mask = explanation.get_image_and_mask(
        label=top_label,
        positive_only=False,   
        num_features=NUM_SEGMENTS,   
        hide_rest=False )
 
    local_exp = explanation.local_exp[top_label]

    
    segment_scores = dict(local_exp)

    importance_map = np.zeros_like(spec_array, dtype=float)
    
    unique_segments = np.unique(segments)
    for seg_id in unique_segments:
        mask = (segments == seg_id)   
        score = segment_scores.get(seg_id, 0.0)   
        importance_map[mask] = score
    
    
    return importance_map, top_label, explanation


def save_lime_results(importance_map, spec_array, song_name, top_label, genre_labels):
    """
    Save the LIME result as .npy and .png
    """
    
    base_path = os.path.join(OUTPUT_DIR, song_name)
    

    np.save(f"{base_path}_lime.npy", importance_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    

    axes[0].imshow(
        spec_array,
        aspect="auto",           
        origin="lower",       
        cmap="viridis"        )
    axes[0].set_title(f"Original Spectrogram\n({song_name})")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Frequency Bin")
    

    im = axes[1].imshow(
        importance_map,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",           
        vmin=-np.abs(importance_map).max(),   
        vmax= np.abs(importance_map).max() )
    plt.colorbar(im, ax=axes[1])
    
    predicted_genre = genre_labels[top_label]
    axes[1].set_title(f"LIME Importance Map\nPredicted: {predicted_genre}")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Frequency Bin")
    
    plt.tight_layout()
    plt.savefig(f"{base_path}_lime.png", dpi=100, bbox_inches="tight")
    plt.close()


print("\n")
print("LIME Explanation Starts")



results_log = []

for idx, row in df.iterrows():
    
   
    filename   = row["filename"]          
    song_type  = row["type"]              
    genre      = row["true_genre"]      
    confidence = row["confidence"]    
    

    song_name = filename.replace(".npy", "")
    
    print(f"\n[{idx+1}/50] {song_name} ({song_type}, {genre}, conf={confidence:.3f})")
    
 
    spec_path = os.path.join(SPEC_DIR, filename)

    
    if not os.path.exists(spec_path):
        print(f" File does not exist: {spec_path}")
        continue
    
    spec_array = np.load(spec_path)
 
    
    if spec_array.ndim == 3:
        spec_array = spec_array[:, :, 0]   
    

    try:
        importance_map, top_label, explanation = generate_lime_for_one_song(
            spec_array, song_name
        )
        

        save_lime_results(
            importance_map, spec_array, song_name, top_label, GENRE_LABELS
        )
        

        results_log.append({
            "filename":    filename,
            "song_name":   song_name,
            "type":        song_type,
            "genre":       genre,
            "confidence":  confidence,
            "top_label":   GENRE_LABELS[top_label],
            "status":      "success"
        })
        
    except Exception as e:
        print(f" error! : {e}")
        results_log.append({
            "filename":   filename,
            "song_name":  song_name,
            "type":       song_type,
            "genre":      genre,
            "confidence": confidence,
            "top_label":  "error",
            "status":     f"error: {e}"
        })


log_df = pd.DataFrame(results_log)
log_path = os.path.join(OUTPUT_DIR, "lime_generation_log.csv")
log_df.to_csv(log_path, index=False)

print("FINISH LIME Explanation")