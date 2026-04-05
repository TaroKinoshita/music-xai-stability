"""
Comparing spectrograms across five genres side by side
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from pathlib import Path
spectrogram_folder = Path(__file__).resolve().parent.parent / "results" / "Spectrogram"

genres = ['classical', 'hiphop', 'jazz', 'metal', 'rock']
file_number = '00000'


def load_spectrogram(genre, file_num):
    filename = f"{genre}.{file_num}.npy"
    filepath = os.path.join(spectrogram_folder, filename)
    
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"No file: {filename}")
        return None

fig, axes = plt.subplots(1, 5, figsize=(26, 5))

for i, genre in enumerate(genres):
    spec = load_spectrogram(genre, file_number)
    
    if spec is not None:
        print(f"  Shape: {spec.shape}")
        print(f"  Range: {spec.min()} ~ {spec.max()}")
        
        im = axes[i].imshow(spec, aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=1)
        
        axes[i].set_title(f'{genre.capitalize()}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time Frames', fontsize=10)
        
        if i == 0:
            axes[i].set_ylabel('Mel Frequency Bins', fontsize=10)
        else:
            axes[i].set_yticks([])

fig.subplots_adjust(right=0.92)  
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
fig.colorbar(im, cax=cbar_ax, label='Normalized Intensity')


fig.suptitle(
    '5 Genres Spectrogram Comparison', 
    fontsize=16, 
    y=0.95
)

output_file = 'genre_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n Done")

plt.close()  
