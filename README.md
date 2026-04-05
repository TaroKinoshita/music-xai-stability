# XAI Music Genre Classification

This is a small project where I trained a CNN on music genre classification and then compared two XAI methods:

- LIME
- Integrated Gradients (IG)

The main question is whether explanations become less stable when the model is uncertain.

---

## Research Question

Does explanation stability differ between **prototypical songs** and **boundary songs** in music genre classification?

In this project:
- **Prototypical songs** = songs the model predicts with high confidence
- **Boundary songs** = songs the model is unsure about, usually close to 50% confidence

---

## Project Structure

```
project/
├── dataset/
│   └── Data/genres_original/   # Due to dataset size, GTZAN should be downloaded manually and create new file in the project folder
├── models/
│   └── cnn_model.h5            # trained CNN model
├── Python/                     # main scripts
│   ├── preprocess.py
│   ├── train_cnn.py
│   ├── select_songs.py
│   ├── IG_Explanation.py
│   ├── LIME_Explanation.py
│   ├── stability_test_.py
│   ├── statistical_analysis.py
│   └── umap_visualization.py
└── results/
    ├── Spectrogram/            # generated spectrograms
    ├── csv/                    # selected songs csv
    ├── IG/                     # IG outputs
    ├── LIME/                   # LIME outputs
    ├── stability/              # CV scores
    ├── statistical/            # plots and summary tables
    └── umap/                   # UMAP figures
```

---

## Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn librosa tqdm lime scikit-image scipy umap-learn
```

---

## Dataset

This project uses the **GTZAN Dataset**. Download it and place it under `dataset/Data/genres_original/`:

> https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Genres used in this project: `classical`, `hiphop`, `jazz`, `metal`, `rock`

---

## How to Run

Run the scripts in this order:

```bash
python Python/preprocess.py
python Python/train_cnn.py
python Python/select_songs.py
python Python/IG_Explanation.py
python Python/LIME_Explanation.py
python Python/stability_test_.py
python Python/statistical_analysis.py
python Python/umap_visualization.py
```

Main outputs will be saved in the `results/` folder.

---

## What Each Script Does

**1. `preprocess.py`**  
Converts `.wav` files into Mel spectrograms and saves them as `.npy` files.  
Output: `results/Spectrogram/`

**2. `train_cnn.py`**  
Trains the CNN on the spectrograms and saves the model.  
Output: `models/cnn_model.h5`, `results/training_history.png`

**3. `select_songs.py`**  
Selects 50 songs for the XAI analysis: 30 prototypical + 20 boundary songs.  
Output: `results/csv/selected_50.csv`

**4. `IG_Explanation.py`**  
Generates Integrated Gradients explanation maps for the selected songs.  
Output: `results/IG/`

**5. `LIME_Explanation.py`**  
Generates LIME explanation maps for the selected songs.  
Output: `results/LIME/`  
Note: This step is slow because LIME uses many random masks.

**6. `stability_test_.py`**  
Runs both explanation methods multiple times with small noise added, then calculates CV to measure stability.  
Output: `results/stability/`

**7. `statistical_analysis.py`**  
Compares prototypical vs boundary songs statistically and generates the main plots.  
Output: `results/statistical/`

**8. `umap_visualization.py`**  
Visualizes the CNN feature space in 2D with UMAP.  
Output: `results/umap/`

---

## Main Idea of the Pipeline

1. Turn audio into spectrograms
2. Train the CNN
3. Choose confident songs and uncertain songs
4. Generate IG and LIME explanations
5. Test how stable those explanations are
6. Compare the two groups statistically
7. Visualize the feature space with UMAP

---

## Main Outputs

Some important output files are:

- `results/training_history.png`  
  shows the CNN training and validation curves

- `results/statistical/fig1_stability_comparison.png`  
  shows the main stability comparison between prototypical and boundary songs

- `results/statistical/fig2_boxplot.png`  
  shows the distribution of CV scores for LIME and IG

- `results/umap/umap_by_genre.png`  
  shows the CNN feature space colored by genre

- `results/umap/umap_by_type.png`  
  shows the same feature space, but colored by song type
---

## Notes

- `IG_Explanation.py` and `LIME_Explanation.py` process all 50 selected songs, so they are not instant.
- `stability_test_.py` is the heaviest script because each song is tested multiple times.
- If you only want the final figures, you do not need to rerun everything from the beginning.
- 
