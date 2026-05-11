import record as rec
import librosa as lr
import spectogram as spec
import numpy as np
import torch
import torch.nn as nn
import model
import queue
import time

RATE = 48000
HOP_LENGTH = 256
DTYPE = np.int16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDX_TO_CHORD = {v: k for k, v in model.CHORD_CLASSES.items()}

window_samples = model.WINDOW_SAMPLES

mode = input("Enter 'record' to capture audio or 'file' to read from a wav file: ").strip().lower()

recording_output_path = "recording_output.wav"

# load model
cnn = model.ChordCNN1D(num_classes=model.NUM_CLASSES, input_bins=84).to(DEVICE)
checkpoint = torch.load("checkpoints/latest.pth", map_location=torch.device(DEVICE))
cnn.load_state_dict(checkpoint['model_state_dict'])
cnn.eval()


def predict_chunk(chunk, cnn_model=cnn, device=DEVICE):
    chunk_spec = spec.Spectogram(chunk, sr=RATE)
    chunk_spec.compute_cqt()
    chunk_spec.cqt_db = (chunk_spec.cqt_db + 80) / 80
    input_tensor = torch.tensor(chunk_spec.cqt_db, dtype=torch.float32).unsqueeze(0).to(device)
    output = cnn_model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    return IDX_TO_CHORD[pred_idx]


# MAIN
if mode == "record":
    recorder = rec.Recording()
    recorder.start_stream()

    audio_buffer = np.array([], dtype=np.float32)
    last_chord = None

    print("Recording... Press Ctrl+C to stop.")
    try:
        with torch.no_grad():
            while True:
                # read available audio chunks from queue
                try:
                    raw_data = recorder.audio_queue.get(timeout=0.05)
                    chunk_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / np.iinfo(DTYPE).max
                    audio_buffer = np.concatenate((audio_buffer, chunk_data))
                except queue.Empty:
                    pass

                # process windows
                while len(audio_buffer) >= window_samples:
                    window = audio_buffer[:window_samples]
                    audio_buffer = audio_buffer[window_samples // 2:]

                    chord_name = predict_chunk(window)
                    if chord_name != last_chord:
                        ts = time.time()
                        #\r moves cursor to beginning line
                        #\033[K clears the line from cursor to end, ensuring old text is erased
                        print(f"\r\033[K{ts:.2f}s Predicted chord: {chord_name}", end="\r")
                        last_chord = chord_name

    except Exception as e:
        print("Error during recording:", e)
    finally:
        recorder.stop_stream()
        recorder.save_recording(recording_output_path)
        print(f"saved recording to {recording_output_path}")

    # combine frames into numpy array
    audio_data = np.frombuffer(b"".join(recorder.frames), dtype=DTYPE)
    audio_float = audio_data.astype(np.float32) / np.iinfo(DTYPE).max

elif mode == "file":
    # load from wav file
    audio_float, _ = lr.load("cruelsummersample.wav", sr=RATE)

    # compute cqt of the full audio for visualization
    spectogram_full = spec.Spectogram(audio_float, sr=RATE)
    spectogram_full.show_spectogram()

    # process audio in WINDOW_SECONDS chunks
    window_samples = model.WINDOW_SAMPLES
    hop_samples = window_samples // 2
    total_samples = len(audio_float)
    predictions = []

    with torch.no_grad():
        offset = 0
        while offset < total_samples:
            chunk = audio_float[offset: offset + window_samples]
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))

            chord_name = predict_chunk(chunk)
            start_sec = offset / RATE
            end_sec = min((offset + window_samples) / RATE, total_samples / RATE)
            predictions.append((start_sec, end_sec, chord_name))

            offset += hop_samples

    # print results (every chord change)
    print(f"\n{'=' * 40}")
    prev_chord = None
    for start, end, chord in predictions:
        if chord != prev_chord:
            print(f"  {start:.2f}s - {end:.2f}s : {chord}")
            prev_chord = chord
    print(f"{'=' * 40}")

else:
    print("Invalid mode")
    exit(1)


