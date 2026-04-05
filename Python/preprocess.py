
"""
Spectrogram Generation Script
load .wav file, convert to spectrogram image. Save as .npy

"""
import os          
import numpy as np 
import librosa    
from tqdm import tqdm 

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
input_folder = ROOT.parent / "dataset" / "Data" / "genres_original"
output_folder = ROOT / "results" / "Spectrogram"


sample_rate = 22050 

duration = 30    
n_mels = 128         

n_fft = 2048         

hop_length = 1024 


def convert_audio_to_image(audio_file_path):
    """
    Convert a single music file into a spectrogram image.
    """
    
    waveform, _ = librosa.load(
        audio_file_path,    
        sr=sample_rate,   
        duration=duration
    )
 
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,         
        sr=sample_rate,     
        n_fft=n_fft,        
        hop_length=hop_length, 
        n_mels=n_mels     
    )
    
    
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    
    min_value = mel_db.min() 
    max_value = mel_db.max() 
    
    normalized = (mel_db - min_value) / (max_value - min_value)
    
    return normalized


def process_all_music_files():
    """
    Convert music files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    genres = [
        'classical',  
        'hiphop',     
        'jazz',       
        'metal',      
        'rock'       ]
    
    
    for genre in genres:
        
        print(f"\n {genre} is processing ---")
        
        genre_folder = os.path.join(input_folder, genre)
        
        all_files = os.listdir(genre_folder)
        
        wav_files = []

        for f in all_files:
            if f.endswith('.wav'):
                wav_files.append(f)
        
        for wav_filename in tqdm(wav_files):
            
            input_path = os.path.join(genre_folder, wav_filename)
            
            
            try:

                spectrogram = convert_audio_to_image(input_path)
                 
                output_filename = wav_filename.replace('.wav', '.npy')
                
                output_path = os.path.join(output_folder, output_filename)
                
                
                # SAve as .npy(NumPy arrays)
                np.save(output_path, spectrogram)
                
            except Exception as error:
                print(f"Error! :  {wav_filename} - {error}")
                continue
    
    
    print(f"\n Process completed ")


if __name__ == "__main__":
    


    process_all_music_files()
    

    processed_files = os.listdir(output_folder)
    npy_files = []

    for f in processed_files:
        if f.endswith('.npy'):
            npy_files.append(f)
    
    print(f"\n Number of files generated: {len(npy_files)}")