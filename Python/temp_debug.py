import numpy as np
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "results" / "Spectrogram"

files = [f for f in os.listdir(SPEC_DIR) if f.endswith(".npy")][:5]
for f in files:
    arr = np.load(os.path.join(SPEC_DIR, f))
    print(f"{f}: shape = {arr.shape}")