## RTDC - Real Time Chord Detection  

RTCD is a compact Python project that performs chord recognition on audio using a trained convolutional neural network (CNN) over Constant-Q Transform (CQT) spectrograms. It supports two modes:

- Real-time (microphone): continuously listens to the default system microphone and prints detected chords to the terminal with low latency.
- File-mode: processes a full audio file and prints a chord timeline after the file is analyzed.

This repository contains the preprocessing, model, and a small runtime script to run inference locally.

**Key characteristics**
- Low-latency streaming using a short sliding window (default 0.25s window, 50% overlap).
- Uses CQT-based spectral features computed with `librosa` and a 1D CNN for classification.
- Minimal dependencies so it can run on CPU or GPU (if PyTorch with CUDA is available).

## Quick Start

1. Clone or open this repository.
2. Create a Python environment (recommended: conda or venv) and install dependencies.

Example (pip):

```bash
python -m pip install --upgrade pip
pip install numpy librosa matplotlib soundfile torch torchvision torchaudio
# PyAudio (for microphone input) can be tricky on some platforms — see Troubleshooting below
```

For PyTorch, follow the instructions on the official site for the best wheel for your platform and CUDA version: https://pytorch.org/get-started/locally/

## Running

From the repository root run:

```bash
python src/recordingTest.py
```

You will be prompted to enter a mode:

- `record` — start capturing from the microphone and output chord predictions in real time. Stop the program with Ctrl+C.
- `file` — process an audio file and print the chord timeline at the end. The program will first show the spectogram of the entire audio file for reference.

Output is printed to the terminal as chords change. Real-time mode prints only when the predicted chord differs from the previous prediction.

## Configuration / Important constants
- Sample rate: 48000 Hz (see `src/model.py` and `src/record.py`).
- Window length: 0.25 seconds (default) — change `WINDOW_SECONDS` in `src/model.py` to adjust latency vs. stability.
- Hop (overlap): code uses 50% overlap (half-window hop) for smoother predictions.

## Project layout
- `src/` — core scripts: audio capture (`record.py`), spectrogram utilities (`spectogram.py`), model definition (`model.py`), dataset helpers, and `recordingTest.py` runtime.
- `checkpoints/` — model checkpoints (place `latest.pth` here for runtime use).
- `data/` — optional dataset and annotation files used during training and evaluation.

## How it works
1. Small segments of audio (default 0.25s) are converted to CQT spectrograms using `librosa`.
2. The spectrogram magnitude is converted to dB and normalized before being fed into the CNN.
3. The model predicts one of the chord classes; the runtime prints changes in prediction as they occur.

## Contributing
Contributions are welcome — open issues for bugs or feature requests. If you change model architecture or training, please add or update a script in `src/` and provide instructions.