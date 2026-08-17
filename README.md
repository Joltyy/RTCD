# RTCD — Real-Time Chord Detection

RTCD recognises musical chords from audio using a 1D convolutional network over
Constant-Q Transform (CQT) spectrograms. It ships as two front ends over one
shared inference engine:

- **Terminal app** (`src/main.py`) — microphone or `.wav` file, prints chords as they change.
- **Web app** (`src/server.py` + `frontend/`) — a React UI streaming **microphone
  *or* browser tab / system audio** to a FastAPI WebSocket backend. Tab audio means
  you can point it at a YouTube video and read the chords live.

Both paths load the same `checkpoints/latest.pth` through `src/chord_engine.py`, so
the terminal app and the web app can never drift apart.

**Current model:** 25 classes (24 major/minor triads + N.C.), **87.87% test accuracy,
0.73 test loss** on a held-out song-level split.

---

## Quick start (terminal app)

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyaudio          # microphone capture only; see Troubleshooting
python src/main.py
```

> `requirements.txt` is the *backend/inference* dependency set and pins CPU-only
> torch. For GPU training, install the CUDA build from
> https://pytorch.org/get-started/locally/ instead.

You'll be prompted for a mode:

- `record` — capture from the default microphone and print predictions in real time.
  Prints only when the chord *changes*. Ctrl+C to stop; the session is saved to
  `recording_output.wav`.
- `file` — analyse a `.wav` file. Shows the full-file spectrogram first, then prints
  the chord timeline.

## Quick start (web app)

Backend, from the repo root:

```bash
pip install -r requirements.txt
cd src
uvicorn server:app --reload --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then pick **Microphone** or **Tab / System Audio** in the UI and press Start. For
tab audio, Chrome only offers the "share tab audio" checkbox when you select a
**tab** (not a window or full screen) in the share picker.

Deployment (Cloud Run backend + Firebase Hosting frontend) is documented
step-by-step in [`FIREBASE_DEPLOY.md`](FIREBASE_DEPLOY.md).

---

## How it works

1. Audio is cut into 0.25 s windows with 50% overlap at inference time.
2. Each window becomes an 84-bin CQT via `librosa`, converted to dB against a
   **fixed** reference (`ref=1.0`), then clipped to a **fixed** floor
   (`DB_FLOOR = -80.0`) and rescaled into `[0, 1]`. Both "fixed"s matter: rescaling
   each window against its own loudest bin would throw away loudness information and
   let training and inference drift apart.
3. The CQT is averaged across the time axis and classified by `ChordCNN1D` into one
   of the 25 classes.
4. Only chord *changes* are emitted — in the terminal, and over the WebSocket.

Sample rate is 48000 Hz throughout. The browser's `AudioContext` is constructed with
`sampleRate: 48000` to match, so no manual resampling is needed anywhere.

### Web audio path

```
mic (getUserMedia) ─┐
                    ├─→ AudioContext(48kHz) → AudioWorkletNode → WebSocket
tab (getDisplayMedia)┘   (public/audio-processor.js buffers to ~4096-sample chunks)
                                    ↓
                    src/server.py  WS /stream
                                    ↓
                    StreamState (sliding window, 50% hop) → ChordEngine.predict()
                                    ↓
                    {time, chord} JSON, pushed only on change
```

`StreamState` exists because the browser sends arbitrarily-sized chunks; it keeps a
running buffer and advances by `hop_samples` so the streaming path reproduces exactly
the sliding-window behaviour `predict_timeline()` gives file mode.

---

## Project layout

```
src/
  main.py            terminal entry point (record / file modes)
  chord_engine.py    ChordEngine — loads the checkpoint, predict() / predict_timeline()
  server.py          FastAPI backend: GET / health check, WS /stream
  model.py           ChordCNN1D, CHORD_CLASSES, ChordSegmentDataset (training segments)
  dataset.py         ChordDataset — on-disk layout, filename conventions, annotation parsing
  spectogram.py      Spectogram — CQT + normalized_cqt(), shared by training and inference
  record.py          Recording — PyAudio mic capture on a background thread
  train.py           training entry point
  librosa_cache.py   sets LIBROSA_CACHE_* env vars; must be imported before librosa
  verify_cache.py    standalone repair tool for a cache interrupted mid-write
frontend/            Vite + React app (see frontend/README.md)
checkpoints/         latest.pth is tracked; epoch_N.pth snapshots are local-only
data/                dataset + data/cache/*.pt segment tensors (gitignored)
Dockerfile           Cloud Run image for src/server.py
firebase.json        Firebase Hosting config (serves frontend/dist)
FIREBASE_DEPLOY.md   full deploy walkthrough
```

### Important constants

| What | Where | Value |
|---|---|---|
| Sample rate | `src/model.py`, `src/record.py` | 48000 Hz |
| Window length | `WINDOW_SECONDS` in `src/model.py` | 0.25 s |
| Hop | inference paths | 50% of window |
| Classes | `CHORD_CLASSES` in `src/model.py` | 25 (24 triads + N.C.) |

Shortening `WINDOW_SECONDS` lowers latency at the cost of prediction stability.

---

## Training

```bash
python src/train.py
```

`train.py` splits **songs** (not windows) into train/test with a fixed seed before any
dataset is built — splitting windows would leak the same song into both sides. Each
song is then chopped into 0.25 s labelled segments and cached as a normalised CQT
tensor under `data/cache/`, with a progress bar showing `new` / `cached` / `segments`
counts.

Caching decodes each song's audio at most **once** and slices segments out of memory,
and writes each tensor via `.tmp` + `os.replace()` so an interrupted run can't leave a
half-written file behind.

Checkpoints: `checkpoints/latest.pth` holds model + optimizer state + epoch/accuracy
and is the single source of truth for both inference and resuming training.
`checkpoints/epoch_N.pth` are historical snapshots.

If a training run is interrupted mid-caching:

```bash
python src/verify_cache.py
```

It scans `data/cache/*.pt`, and quarantines anything that fails to load (plus stray
`*.pt.tmp` files) into `_to_delete/corrupt_cache/` so the next run regenerates only
those segments rather than the whole cache.

> **Note on `.librosa_cache/`:** `LIBROSA_CACHE_LEVEL` is deliberately pinned to `10`
> in `src/librosa_cache.py`. Level 10 caches only librosa's audio-*independent* filter
> bank construction. Higher levels cache functions that take the actual audio as an
> argument — those entries are never reused and the cache grows without bound (it once
> reached ~90 GB). Don't raise it.

---

## Roadmap

Done: standalone inference engine, song-level train/test split, fixed-reference
normalisation, deployed web app, tab/system audio capture.

Next, in rough priority order:

1. **Accuracy** — pitch-shift augmentation and a factored root/quality head first
   (cheapest, highest impact), then temporal modelling (CRNN) instead of averaging
   across the time axis, HPSS preprocessing, and prediction smoothing. Everything now
   has an 87.87% baseline to be measured against.
2. **Features** — key detection, BPM/tempo, beat-synced smoothing; further out,
   Roman-numeral analysis, chord diagrams, section detection, chart export.
3. **Deferred web work** — file-upload mode (`POST /analyze-file`), Firebase Auth +
   Firestore session history, tightening backend CORS (currently `allow_origins=["*"]`),
   and a real YouTube Data API search if the share-a-tab round trip gets annoying.

A note on complex chords: a sample of the `_beatinfo.arff` annotations contained only
major/minor triads. Supporting 7ths, dim, aug, sus etc. likely needs a richer
annotation source or a different dataset — not just a bigger `CHORD_CLASSES` dict.

---

## Troubleshooting

**PyAudio won't install.** It needs PortAudio headers. On Windows, prefer a prebuilt
wheel (`pip install pipwin && pipwin install pyaudio`); on macOS `brew install
portaudio` first; on Debian/Ubuntu `sudo apt install portaudio19-dev`. PyAudio is only
needed for terminal `record` mode — file mode and the whole web app work without it.

**Tab audio capture gives "no audio track".** You selected a window or entire screen
instead of a tab, or left "Share tab audio" unchecked. Only the tab option exposes
audio in Chrome.

**Cloud Run container fails to start.** `model.py` is shared by the training and
serving paths and imports everything at module load — including `tqdm`, which the
server never calls. Any top-level import in `model.py`, `dataset.py`, or
`spectogram.py` must be in `requirements.txt`, not just the ones the serving path
exercises. Read the real traceback with
`gcloud run services logs read chordthingy-backend --region=us-central1`; Cloud Run's
generic "failed to start and listen on port" message looks identical for a slow cold
start and an outright crash.
