import os
import torch
import torch.nn as nn
import librosa as lr
import numpy as np
import spectogram
import dataset
from torch.utils.data import Dataset, DataLoader

RATE = 48000
HOP_LENGTH = 256
WINDOW_SECONDS = 0.25 # Increasing this will add latency
WINDOW_FRAMES = int(WINDOW_SECONDS * RATE / HOP_LENGTH)
WINDOW_SAMPLES = int(WINDOW_SECONDS * RATE)

CHORD_CLASSES = {
    "Cmaj": 0, "Cmin": 1,
    "C#maj": 2, "C#min": 3,
    "Dmaj": 4, "Dmin": 5,
    "D#maj": 6, "D#min": 7,
    "Emaj": 8, "Emin": 9,
    "Fmaj": 10, "Fmin": 11,
    "F#maj": 12, "F#min": 13,
    "Gmaj": 14, "Gmin": 15,
    "G#maj": 16, "G#min": 17,
    "Amaj": 18, "Amin": 19,
    "A#maj": 20, "A#min": 21,
    "Bmaj": 22, "Bmin": 23,
    "N.C.": 24
}
NUM_CLASSES = len(CHORD_CLASSES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#PREPROCESSING
class ChordSegmentDataset(Dataset):
    def __init__(self, raw_dataset, cache_dir="data\cache"):
        self.raw_dataset = raw_dataset
        self.segments = [] # (file idx, start sample, label idx)
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        print("preprocessing data...")
        for i in range(len(self.raw_dataset.audio_dir)):
            audio_path = self.raw_dataset.audio_dir[i]
            file_id = dataset.os.path.basename(audio_path).split('_')[0] # ex. 0001 from 0001_mix.flac
            annot_path = dataset.os.path.join(self.raw_dataset.annotation_dir, f"{file_id}_beatinfo.arff")

            #get duration of the audio file in seconds
            duration = lr.get_duration(filename=audio_path, sr=RATE)

            #(start_time, end_time, chord_label) for each chord event in the annotation file
            events = self.raw_dataset.parseBeatInfo(annot_path, duration)

            #generate segemnts of WINDOW_SECONDS length for each chord event to be the data points for training
            for start_t, end_t, label in events:
                curr_time = start_t
                while curr_time < end_t:
                    duration_remaining = min(WINDOW_SECONDS, end_t - curr_time)
                    start_sample = int(curr_time * RATE)
                    cache_path = os.path.join(self.cache_dir, f"{file_id}_{start_sample}.pt")

                    #precompute and cache
                    if not os.path.exists(cache_path): #if not yet loaded, load and cache it
                        y, _ = lr.load(audio_path, sr=RATE, offset=curr_time, duration=duration_remaining)
                        if len(y) < WINDOW_SAMPLES:
                            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
                        else:
                            y = y[:WINDOW_SAMPLES]
                        
                        #compute spectogram and save as tensor for faster loading during training
                        spec = spectogram.Spectogram(y, sr=RATE)
                        spec.compute_cqt()
                        spec.cqt_db = (spec.cqt_db + 80) / 80
                        tensor = torch.tensor(spec.cqt_db, dtype=torch.float32).unsqueeze(0)
                        torch.save(tensor, cache_path)

                    #save segment metadata, actual dat will be loaded from cache during training
                    self.segments.append((cache_path, CHORD_CLASSES[label]))
                    curr_time += WINDOW_SECONDS

        print(f"generated {len(self.segments)} training segments.")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        cache_path, label_idx = self.segments[idx]
        spec_tensor = torch.load(cache_path, weights_only=True)
        
        #Ensure shape is [Freq, Time]
        while spec_tensor.dim() > 2:
            spec_tensor = spec_tensor.squeeze(0)
        
        return spec_tensor, torch.tensor(label_idx, dtype=torch.long)


#MODEL
class ChordCNN1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_bins=84):
        super(ChordCNN1D, self).__init__()
        
        self.features = nn.Sequential(
            # Input: [Batch, 1, 84]
            
            # Layer 1
            nn.Conv1d(1, 64, kernel_size=5, padding=2), # Look at neighbors 5 semitones wide
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 84 -> 42 bins
            
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), # 42 -> 21 bins
            
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2), # 21 -> 10 bins
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (input_bins // 8), 256), # approx 256 * 10 = 2560 inputs
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape from DataLoader: [Batch, Freq, Time]
        
        # Average across time (last dim)
        x = x.mean(dim=-1)      # [Batch, Freq]
        x = x.unsqueeze(1)      # [Batch, 1, Freq]  — channel dim for Conv1d
        
        x = self.features(x)
        x = self.classifier(x)
        
        return x