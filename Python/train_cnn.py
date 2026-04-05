
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spectrogram_folder = ROOT / "results" / "Spectrogram"
model_save_path = ROOT / "models" / "cnn_model.h5"
results_folder = ROOT / "results"

TARGET_HEIGHT = 128
TARGET_WIDTH = 677

genres = ['classical', 'hiphop', 'jazz', 'metal', 'rock']

EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

def resize_to_fixed_size(spectrogram):
    """
    Make all spectrograms the same size: (128, 677)
    
    If too short, add zeros
    If too long,cut from center
    """
    height, width = spectrogram.shape
    
    if width > TARGET_WIDTH:
        start = (width - TARGET_WIDTH) // 2
        return spectrogram[:, start:start + TARGET_WIDTH]
    
    elif width < TARGET_WIDTH:
        pad_needed = TARGET_WIDTH - width
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        return np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), constant_values=0)
    
    else:
        return spectrogram

def load_data():
    """
    Load all .npy files from spectrogram folder
    """

    files = [f for f in os.listdir(spectrogram_folder) if f.endswith('.npy')]
    print(f"Found {len(files)} files")
    
    spectrograms = []
    labels = []
    
    for i, filename in enumerate(files):
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")
        
        filepath = os.path.join(spectrogram_folder, filename)
        spec = np.load(filepath)
        
        spec_resized = resize_to_fixed_size(spec)
        
        genre_name = filename.split('.')[0]
        label = genres.index(genre_name)
        
        spectrograms.append(spec_resized)
        labels.append(label)
    
    X = np.array(spectrograms)
    y = np.array(labels)
    
    return X, y

def build_model():
    """
    Build 5-layer CNN for genre classification
    """
    
    model = models.Sequential([
        
        layers.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
    
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN model
    """
    print("Training model")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    
    early_stop = EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True,verbose=1)
    
    checkpoint = ModelCheckpoint(model_save_path,monitor='val_accuracy',save_best_only=True,verbose=1)
    
    history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[early_stop, checkpoint],verbose=1)

    
    return history

def plot_history(history):
    """
    Create training history plot
    Training vs validation accuracy
    Training vs validation loss
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_folder, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"saved: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    
    X, y = load_data()
    
    X = X[..., np.newaxis]  
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=5)
    
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=VALIDATION_SPLIT,random_state=67,stratify=np.argmax(y, axis=1) )
    
    model = build_model()
    
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    plot_history(history)


    print("FINAL result")

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"Final training accuracy:   {final_train_acc}")
    print(f"Final validation accuracy:  {final_val_acc}")
    print(f"Best validation accuracy:     {best_val_acc}")
    
    if best_val_acc >= 0.60:
        print("\n SUCCESS")
    else:
        print(f"\n Model accuracy is below target ({best_val_acc} < 60%)")
    
