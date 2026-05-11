
# Convert audio data in raw byte format into spectograms

import numpy as np
import matplotlib.pyplot as plt
import librosa as lr

class Spectogram:
    def __init__(self, audio_data, sr=48000):
        self.audio_data = audio_data
        self.sr = sr
        self.cqt_spectrum = None
        self.cqt_db = None
        self.chroma = None
    
    def compute_cqt(self):
        # apply cqt to the audio data to get the spectogram
        # lr.cqt returns a complex data so we take the absolute
        # value to get the magnitude of the frequency
        self.cqt_spectrum = np.abs(lr.cqt(y=self.audio_data, sr=self.sr))
        self.cqt_db = lr.amplitude_to_db(self.cqt_spectrum, ref=np.max)
    
    def compute_chroma(self):
        self.chroma = lr.amplitude_to_db(lr.feature.chroma_cqt(y=self.audio_data, sr=self.sr), ref=np.max)
    
    def show_spectogram(self, chord_events=None, use_chroma=False):
        if use_chroma:
            if self.chroma is None:
                    self.compute_chroma()
            data = self.chroma
            y_axis_type = 'chroma'
            title = 'Chromagram'
        else:
            if self.cqt_db is None:
                self.compute_cqt()
            data = self.cqt_db
            y_axis_type = 'cqt_note'
            title = 'Constant-Q Power Spectrogram'
        
        plt.figure(figsize=(10, 6))
        lr.display.specshow(data, sr=self.sr, x_axis='time', y_axis=y_axis_type, cmap='coolwarm')

        # show the chords aswell if available
        if chord_events:
            for start, end, label in chord_events:
                # Draw vertical line at the start of the chord
                plt.axvline(x=start, color='white', linestyle=':', alpha=0.6)
                
                # Place text in the middle of the duration, just below the x-axis
                mid_time = (start + end) / 2
                plt.text(mid_time, -0.02, label, color='black', rotation=90, 
                         transform=plt.gca().get_xaxis_transform(), 
                         ha='center', va='top', fontsize=8)
                
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Power Spectrogram')
        plt.tight_layout()
        plt.show()