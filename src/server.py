import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import librosa_cache

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from chord_engine import ChordEngine

CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "checkpoints", "latest.pth")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ChordEngine(checkpoint_path=CHECKPOINT_PATH)


@app.get("/")
def health_check():
    return {"status": "ok", "device": engine.device, "rate": engine.rate}


class StreamState:
    """
    Turns a sequence of arbitrarily-sized incoming audio chunks into a
    sliding-window stream of chord predictions.
    """

    def __init__(self, engine: ChordEngine):
        self.engine = engine
        self.window_samples = engine.window_samples
        self.hop_samples = self.window_samples // 2
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_consumed = 0
        self.last_chord = None

    def push(self, chunk: np.ndarray):
        """Feed in new samples, return a list of (time_sec, chord) results
        for every window completed by this push (usually 0 or 1, but could
        be more than one if a single incoming chunk is unusually large)."""
        self.buffer = np.concatenate([self.buffer, chunk])
        results = []
        while len(self.buffer) >= self.window_samples:
            window = self.buffer[: self.window_samples]
            chord = self.engine.predict(window)
            t = self.samples_consumed / self.engine.rate
            results.append((t, chord))

            self.buffer = self.buffer[self.hop_samples :]
            self.samples_consumed += self.hop_samples
        return results


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    state = StreamState(engine)
    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)

            for t, chord in state.push(chunk):
                if chord != state.last_chord:
                    state.last_chord = chord
                    await websocket.send_json({"time": t, "chord": chord})
    except WebSocketDisconnect:
        pass
