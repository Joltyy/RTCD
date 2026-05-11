import librosa
import numpy as np
import os
import glob

class ChordDataset:
    def __init__(self, audio_dir, annotation_dir, sr, hop_length):
        self.audio_dir = sorted(glob.glob(os.path.join(audio_dir, "*.flac")))
        self.annotation_dir = annotation_dir
        self.sr = sr
        self.hop_length = hop_length
    
    def parseBeatInfo(self, annotation_path, total_duration):
        events = []

        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        # get the lines that are not starting with '@' and are not empty
        data_lines = [l.strip() for l in lines if not l.startswith('@')
                      and l.strip() != '']
        
        for i, line in enumerate(data_lines):
            # format: 0.0,1,1,'Fmaj'
            parts = line.split(',')
            start_time = float(parts[0])
            chord = parts[3].strip().strip("'")

            if i < len(data_lines) - 1:
                next_start_time = float(data_lines[i + 1].split(',')[0])
                end_time = next_start_time
            else:
                #for last chord, the end_time should be the end of the audio file
                end_time = total_duration
            
            events.append((start_time, end_time, chord))

        return events

    def getItem(self, idx):
        audio_path = self.audio_dir[idx] # format: xxxx_mix.flac
        file_id = os.path.basename(audio_path).split('_')[0] # gets the base number ex. '0001' from '0001_mix.flac'
        
        annotation_filename = os.path.join(self.annotation_dir, f"{file_id}_beatinfo.arff")

        y, _ = librosa.load(audio_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=self.sr)

        chord_events = self.parseBeatInfo(annotation_filename, duration)

        return y, chord_events
        