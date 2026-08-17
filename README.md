# RTCD — Real-Time Chord Detection

Recognises musical chords from audio in real time, using a 1D convolutional network
over Constant-Q Transform (CQT) spectrograms.

There are two ways to run it, both sharing one inference engine:

- **Terminal app** — microphone or a `.wav` file, prints chords as they change.
- **Web app** — a React UI that streams **microphone _or_ browser tab / system audio**
  to a FastAPI WebSocket backend. Tab audio means you can point it at a YouTube video
  and read the chords live.

**Model:** 25 classes (24 major/minor triads + N.C.), 87.87% test accuracy on a
held-out song-level split. Currently triads only — no 7ths, dim, aug or sus.

---

## Requirements

- Python 3.10+ (3.11 is what the Docker image uses)
- Node 18+ and npm, for the web frontend
- A trained checkpoint at `checkpoints/latest.pth` — one is included in the repo

## Install

```bash
git clone <this-repo>
cd chordthingy

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` pins **CPU-only** torch, which is all you need for inference. If
you want to train on a GPU, install the CUDA build from
[pytorch.org](https://pytorch.org/get-started/locally/) instead.

For terminal microphone mode you also need PyAudio, which isn't in
`requirements.txt` because it needs system libraries:

```bash
pip install pyaudio
```

If that fails, see [Troubleshooting](#troubleshooting). File mode and the entire web
app work fine without it.

---

## Running the terminal app

```bash
python src/main.py
```

You'll be prompted for a mode:

| Mode | What it does |
|---|---|
| `record` | Listens to the default microphone and prints predictions live. Prints only when the chord *changes*. Ctrl+C to stop; the session is saved to `recording_output.wav`. |
| `file` | Analyses a `.wav` file. Shows the full-file spectrogram, then prints the chord timeline. |

## Running the web app

Two terminals. Backend first, from the repo root:

```bash
cd src
uvicorn server:app --reload --port 8000
```

Then the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`), pick **Microphone** or
**Tab / System Audio**, and press Start.

No environment file is needed for local dev — the frontend falls back to
`ws://<hostname>:8000/stream` when `VITE_BACKEND_WS_URL` is unset.

> **Tab audio tip:** Chrome only offers the "Share tab audio" checkbox when you select
> a **tab** in the share picker. Picking a window or your entire screen gives you a
> stream with no audio track, and the app will tell you so rather than silently
> connecting to nothing.

See [`frontend/README.md`](frontend/README.md) for frontend specifics, and
[`FIREBASE_DEPLOY.md`](FIREBASE_DEPLOY.md) for deploying to Cloud Run + Firebase
Hosting.

---

## How it works

1. Audio is cut into 0.25 s windows with 50% overlap.
2. Each window becomes an 84-bin CQT via `librosa`, converted to dB against a fixed
   reference (`ref=1.0`), clipped to a fixed floor (`-80 dB`), and rescaled to
   `[0, 1]`. Both references are *fixed* on purpose — normalising each window against
   its own loudest bin would discard loudness information and let training and
   inference drift apart.
3. The CQT is averaged across the time axis and classified by `ChordCNN1D` into one
   of the 25 classes.
4. Only chord *changes* are emitted, in the terminal and over the WebSocket alike.

Everything runs at 48 kHz. The browser's `AudioContext` is constructed with
`sampleRate: 48000` to match, so no manual resampling happens anywhere.

### Web audio path

```
mic (getUserMedia) ─┐
                    ├─→ AudioContext(48kHz) → AudioWorkletNode → WebSocket
tab (getDisplayMedia)┘   (buffers to ~4096-sample chunks)
                                    ↓
                         src/server.py   WS /stream
                                    ↓
                    StreamState (sliding window, 50% hop)
                                    ↓
                         ChordEngine.predict()
                                    ↓
                    {time, chord} JSON, sent only on change
```

`StreamState` exists because browsers send arbitrarily-sized chunks. It keeps a
running buffer and advances by one hop at a time, so the streaming path reproduces
exactly the sliding-window behaviour that file mode gets from `predict_timeline()`.

---

## Project layout

```
src/
  main.py            terminal entry point (record / file modes)
  chord_engine.py    ChordEngine — loads the checkpoint, predict() / predict_timeline()
  server.py          FastAPI backend: GET / health check, WS /stream
  model.py           ChordCNN1D, CHORD_CLASSES, ChordSegmentDataset
  dataset.py         ChordDataset — on-disk layout, filenames, annotation parsing
  spectogram.py      Spectogram — CQT + normalized_cqt(), shared by training and inference
  record.py          Recording — PyAudio mic capture on a background thread
  train.py           training entry point
  librosa_cache.py   sets LIBROSA_CACHE_* env vars; imported before librosa everywhere
  verify_cache.py    repair tool for a segment cache interrupted mid-write
frontend/            Vite + React app
checkpoints/         latest.pth (tracked); epoch_N.pth snapshots (local only)
data/                dataset + cached segment tensors (gitignored)
Dockerfile           Cloud Run image for src/server.py
firebase.json        Firebase Hosting config
```

### Key constants

| What | Where | Value |
|---|---|---|
| Sample rate | `src/model.py`, `src/record.py` | 48000 Hz |
| Window length | `WINDOW_SECONDS` in `src/model.py` | 0.25 s |
| Hop | inference paths | 50% of window |
| dB floor | `DB_FLOOR` in `src/spectogram.py` | -80.0 |
| Classes | `CHORD_CLASSES` in `src/model.py` | 25 |

Shortening `WINDOW_SECONDS` lowers latency at the cost of prediction stability.

---

## Training

You only need this if you want to retrain — a working checkpoint ships with the repo.

The expected dataset layout is `<id>_mix.flac` audio paired with `<id>_beatinfo.arff`
annotations, in two sibling directories. The paths are currently **hardcoded** near
the top of `src/train.py`:

```python
audio_path = os.path.join("data", "0001-1000-audio-mixes_2")
annot_path = os.path.join("data", "0001-1000-annotations-v1.1.0")
```

Edit those to point at your own data, then run from the repo root:

```bash
python src/train.py
```

Songs (not windows) are split into train/test with a fixed seed *before* any dataset
is built — splitting windows would leak the same song into both sides. Each song is
then chopped into 0.25 s labelled segments and cached as a CQT tensor under
`data/cache/`.

The first run is slow because it builds that cache; later runs reuse it. Writes are
atomic, so an interrupted run can't leave a half-written tensor behind. If a run *is*
interrupted mid-caching:

```bash
python src/verify_cache.py
```

That quarantines any unreadable cache entries so the next run regenerates only those,
rather than the whole cache.

`checkpoints/latest.pth` holds model + optimizer state + epoch/accuracy and is what
both inference and training-resume read.

> **Don't raise `LIBROSA_CACHE_LEVEL`.** It's pinned to `10` in
> `src/librosa_cache.py`, which caches only librosa's audio-*independent* filter bank
> construction. Higher levels also cache functions that take the audio itself as an
> argument — those entries are never reused, and the cache grows without bound.

---

## Troubleshooting

**PyAudio won't install.** It needs PortAudio headers:

- Windows: `pip install pipwin && pipwin install pyaudio`
- macOS: `brew install portaudio` first
- Debian/Ubuntu: `sudo apt install portaudio19-dev` first

Only terminal `record` mode needs it.

**"No audio track" when sharing a tab.** You selected a window or your entire screen
instead of a tab, or left "Share tab audio" unchecked. Only the tab option exposes
audio in Chrome.

**`ModuleNotFoundError` in the deployed container.** `model.py` is shared by the
training and serving paths and imports everything at module load, including packages
the server never calls. Any top-level import in `model.py`, `dataset.py`, or
`spectogram.py` has to be in `requirements.txt` — not just the ones serving actually
exercises.

---

## Contributing

Issues and PRs welcome. A few things worth knowing:

- There's no automated test suite yet. If you touch the frontend's session lifecycle
  (`useChordStream.js`), please test start/stop races by hand — see
  `frontend/README.md` for the specific failure modes that have bitten before.
- `src/spectogram.py`'s `normalized_cqt()` is shared by training and inference. Change
  it and you invalidate `data/cache/` and every existing checkpoint.
- Run `npm run lint` and `npm run build` in `frontend/` before submitting.
