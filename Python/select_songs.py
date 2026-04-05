"""
Select 50 songs: 30 prototypical + 20 boundary
Lower threshold if not enough high-confidence songs
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
model_path = ROOT / "models" / "cnn_model.h5"
spectrogram_folder = ROOT / "results" / "Spectrogram"
output_csv = ROOT / "results" / "csv" / "selected_50.csv"

TARGET_WIDTH = 677
genres = ['classical', 'hiphop', 'jazz', 'metal', 'rock']

BOUNDARY_MIN = 0.45
BOUNDARY_MAX = 0.55
PROTOTYPICAL_PER_GENRE = 6
BOUNDARY_TOTAL = 20


def resize_spectrogram(spec):
    height, width = spec.shape
    
    if width > TARGET_WIDTH:
        start = (width - TARGET_WIDTH) // 2
        return spec[:, start:start + TARGET_WIDTH]
    elif width < TARGET_WIDTH:
        pad = TARGET_WIDTH - width
        return np.pad(spec, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=0)
    else:
        return spec

def predict_all(model):
    files = [f for f in os.listdir(spectrogram_folder) if f.endswith('.npy')]
    results = []
    
    for filename in files:
        spec = np.load(os.path.join(spectrogram_folder, filename))
        spec = resize_spectrogram(spec)
        spec_input = spec[np.newaxis, ..., np.newaxis]
        
        pred = model.predict(spec_input, verbose=0)[0]
        
        true_genre = filename.split('.')[0]
        predicted_genre = genres[np.argmax(pred)]
        confidence = np.max(pred)
        
        top2_ids = np.argsort(pred)[-2:][::-1]
        top2_genres = f"{genres[top2_ids[0]]}-{genres[top2_ids[1]]}"
        top2_probs = f"{pred[top2_ids[0]]:.2f}-{pred[top2_ids[1]]:.2f}"
        
        results.append({'filename': filename,'true_genre': true_genre,'predicted_genre': predicted_genre,
            'confidence': confidence,
            'top2_genres': top2_genres,
            'top2_probs': top2_probs
        })
    
    return results


def select_songs(results):
    selected = []
    
    for genre in genres:
        genre_songs = [
            r for r in results 
            if r['true_genre'] == genre and r['predicted_genre'] == genre
        ]
        
        genre_songs.sort(key=lambda x: x['confidence'], reverse=True)
        
        for song in genre_songs[:PROTOTYPICAL_PER_GENRE]:
            song['type'] = 'prototypical'
            selected.append(song)
    
    boundary = [r for r in results if BOUNDARY_MIN <= r['confidence'] <= BOUNDARY_MAX]
    boundary.sort(key=lambda x: abs(x['confidence'] - 0.50))
    
    for song in boundary[:BOUNDARY_TOTAL]:
        song['type'] = 'boundary'
        selected.append(song)
    
    return selected

if __name__ == "__main__":
    
    model = load_model(model_path)
    
    results = predict_all(model)
    
    selected = select_songs(results)
    
    df = pd.DataFrame(selected)
    df = df[['filename', 'true_genre', 'predicted_genre', 'confidence', 'type', 'top2_genres', 'top2_probs']]
    df = df.sort_values(['type', 'true_genre', 'confidence'], ascending=[False, True, False])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    proto = df[df['type'] == 'prototypical']
    boundary = df[df['type'] == 'boundary']
    
    print(f"\n Done: {len(df)} songs")
    print(f"\nPrototypical: {len(proto)}")
    for genre in genres:
        count = len(proto[proto['true_genre'] == genre])
        avg_conf = proto[proto['true_genre'] == genre]['confidence'].mean()
        print(f"  {genre}: {count} songs (avg: {avg_conf:.2%})")
    
    print(f"\nBoundary: {len(boundary)} (avg: {boundary['confidence'].mean():.2%})")