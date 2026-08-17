import librosa_cache  # noqa: F401  -- must be imported before `librosa` itself (see librosa_cache.py)

import os
import torch
import torch.nn as nn
import librosa as lr
import numpy as np
import spectogram
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

RATE = 48000
WINDOW_SECONDS = 0.25 # Increasing this will add latency
WINDOW_SAMPLES = int(WINDOW_SECONDS * RATE)
# NOTE: there used to be a HOP_LENGTH=256 / WINDOW_FRAMES constant here.
# Removed: nothing in the actual pipeline ever passed HOP_LENGTH into
# librosa.cqt() (spectogram.py's compute_cqt() calls lr.cqt() with no
# hop_length, so it silently used librosa's own default of 512 the whole
# time), so those two constants never affected behavior -- they were dead
# weight left over from an earlier version of the code.

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
    # NOTE: this was briefly bumped to "data/cache_v2" right after the CQT
    # normalization changed (see Spectogram.normalized_cqt()), specifically
    # so stale old-normalization tensors couldn't get silently mixed in with
    # freshly-computed ones under the same folder name. Since both old
    # cache folders have now been deleted, there's nothing left to collide
    # with, so this is back to the plain "data/cache" name.
    def __init__(self, raw_dataset, cache_dir=os.path.join("data", "cache")):
        self.raw_dataset = raw_dataset
        self.segments = [] # (file idx, start sample, label idx)
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        print("preprocessing data...")
        # tqdm wraps the per-SONG loop (that's the natural unit of progress --
        # "song 42/1000"), but the actually expensive work (audio decode + CQT)
        # happens per-SEGMENT inside it, and a lot of segments might already be
        # cached from a previous run. set_postfix() below surfaces those two
        # numbers live so "is this actually doing anything" is visible even
        # while a single song is still in progress, not just between songs.
        newly_cached = 0
        already_cached = 0
        song_progress = tqdm(
            self.raw_dataset.audio_dir, desc="Caching segments", unit="song"
        )
        for audio_path in song_progress:
            # file_id / annotation-path naming convention now lives in ONE
            # place (dataset.ChordDataset), instead of being duplicated here
            # via the odd `dataset.os.path...` cross-module reach into
            # dataset.py's own `os` import.
            file_id = self.raw_dataset.file_id_for(audio_path)
            annot_path = self.raw_dataset.annotation_path_for(audio_path)

            #get duration of the audio file in seconds
            duration = lr.get_duration(path=audio_path, sr=RATE)

            #(start_time, end_time, chord_label) for each chord event in the annotation file
            events = self.raw_dataset.parseBeatInfo(annot_path, duration)

            # Decode this song's audio at most ONCE, lazily -- only if at
            # least one of its segments actually needs (re)computing.
            # Previously, every single segment called lr.load(audio_path,
            # offset=, duration=), which reopens and re-seeks into the same
            # FLAC file from scratch for every ~0.25s window -- easily
            # hundreds of separate file opens per song. A fully-cached song
            # (the common case on resume) still never touches the audio
            # file at all here, keeping resume fast; a song with any
            # missing segments now pays for exactly one decode of the whole
            # song instead of one decode per segment.
            full_audio = None

            #generate segemnts of WINDOW_SECONDS length for each chord event to be the data points for training
            for start_t, end_t, label in events:
                curr_time = start_t
                while curr_time < end_t:
                    duration_remaining = min(WINDOW_SECONDS, end_t - curr_time)
                    start_sample = int(curr_time * RATE)
                    cache_path = os.path.join(self.cache_dir, f"{file_id}_{start_sample}.pt")

                    #precompute and cache
                    if not os.path.exists(cache_path): #if not yet loaded, load and cache it
                        if full_audio is None:
                            full_audio, _ = lr.load(audio_path, sr=RATE)

                        segment_samples = int(duration_remaining * RATE)
                        y = full_audio[start_sample : start_sample + segment_samples]
                        if len(y) < WINDOW_SAMPLES:
                            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
                        else:
                            y = y[:WINDOW_SAMPLES]

                        #compute spectogram and save as tensor for faster loading during training
                        #(normalized_cqt() is the SAME normalization used at inference time in
                        # chord_engine.py -- keeping this in one shared place avoids training and
                        # inference silently drifting apart)
                        spec = spectogram.Spectogram(y, sr=RATE)
                        normalized = spec.normalized_cqt()
                        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

                        # Write atomically: save to a temp file in the same
                        # directory, then rename it into place only once the
                        # save has fully succeeded. os.replace() is an atomic
                        # rename on both Windows and POSIX, so cache_path can
                        # only ever exist as a COMPLETE, valid file. If the
                        # write is interrupted (crash, disk full, power loss),
                        # the .tmp file might end up corrupt, but cache_path
                        # itself is never touched -- so a later run correctly
                        # sees "not cached yet" and regenerates it, instead of
                        # silently treating a truncated file as done and
                        # blowing up later during actual training.
                        tmp_path = cache_path + ".tmp"
                        try:
                            torch.save(tensor, tmp_path)
                            os.replace(tmp_path, cache_path)
                        except Exception:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            raise
                        newly_cached += 1
                    else:
                        already_cached += 1

                    #save segment metadata, actual dat will be loaded from cache during training
                    self.segments.append((cache_path, CHORD_CLASSES[label]))
                    curr_time += WINDOW_SECONDS

            song_progress.set_postfix(
                segments=len(self.segments), new=newly_cached, cached=already_cached
            )

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