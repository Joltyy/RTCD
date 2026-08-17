import librosa_cache

import numpy as np
import torch

import model
import spectogram


class ChordEngine:
    def __init__(self, checkpoint_path="checkpoints/latest.pth", device=None):
        self.device = device or model.DEVICE
        self.window_samples = model.WINDOW_SAMPLES
        self.rate = model.RATE
        self.idx_to_chord = {v: k for k, v in model.CHORD_CLASSES.items()}

        self.model = model.ChordCNN1D(
            num_classes=model.NUM_CLASSES, input_bins=84
        ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, chunk: np.ndarray) -> str:
        """
        Predict a single chord label for one audio chunk.
        """
        chunk = self._fit_to_window(chunk)

        spec = spectogram.Spectogram(chunk, sr=self.rate)
        features = spec.normalized_cqt()

        input_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            output = self.model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return self.idx_to_chord[pred_idx]

    def predict_timeline(self, audio: np.ndarray, hop_samples=None):
        """
        Slide the window over a full audio array (file mode) and yield
        (start_sec, end_sec, chord) for every window, in order. This is a
        generator so a caller (CLI, web endpoint, etc.) can consume results
        as they're produced instead of waiting for the whole file.
        """
        hop = hop_samples or self.window_samples // 2
        total_samples = len(audio)
        offset = 0
        while offset < total_samples:
            chunk = audio[offset : offset + self.window_samples]
            chord = self.predict(chunk)
            start_sec = offset / self.rate
            end_sec = min(
                (offset + self.window_samples) / self.rate, total_samples / self.rate
            )
            yield start_sec, end_sec, chord
            offset += hop

    def _fit_to_window(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) < self.window_samples:
            return np.pad(chunk, (0, self.window_samples - len(chunk)))
        if len(chunk) > self.window_samples:
            return chunk[: self.window_samples]
        return chunk
