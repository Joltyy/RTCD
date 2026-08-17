import librosa_cache  # noqa: F401  -- must be imported before `librosa` itself (see librosa_cache.py)

import record as rec #contains the Recording class for capturing audio from the microphone
import librosa as lr
import spectogram as spec #contains the main spectogram plotting and librosa operations
import numpy as np
import queue
import time

from chord_engine import ChordEngine #main class for chord recognition

RATE = 48000
DTYPE = np.int16

mode = input("Enter 'record' to capture audio or 'file' to read from a wav file: ").strip().lower()
file_path = input("Enter the path to the wav file: ").strip() if mode == "file" else None

recording_output_path = "recording_output.wav"

# all model loading / spectrogram / normalization details now live in
# ChordEngine (see chord_engine.py) -- this script just drives it
engine = ChordEngine(checkpoint_path="checkpoints/latest.pth")
window_samples = engine.window_samples


# MAIN
if mode == "record":
    recorder = rec.Recording()
    recorder.start_stream()

    audio_buffer = np.array([], dtype=np.float32)
    last_chord = None

    print("Recording... Press Ctrl+C to stop.")
    try:
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

                chord_name = engine.predict(window)
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

elif mode == "file":
    # load from wav file
    audio_float, _ = lr.load(file_path, sr=RATE)

    # compute cqt of the full audio for visualization
    spectogram_full = spec.Spectogram(audio_float, sr=RATE)
    spectogram_full.show_spectogram()

    # process audio in WINDOW_SECONDS chunks, via the engine's sliding window
    predictions = list(engine.predict_timeline(audio_float))

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
