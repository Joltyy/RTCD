# RTCD frontend

React (Vite) app for the real-time chord detector. Captures audio from either the
**microphone** or a **browser tab / system audio**, streams it to the Python backend
over a WebSocket, and displays predicted chords live.

Tab audio is what makes it possible to read chords off a YouTube video: play the song
in another tab, share that tab back through the browser's own picker, and the audio
flows through the identical pipeline the mic uses.

## Local dev

You need the backend running first (from the repo root):

```bash
cd ../
pip install -r requirements.txt   # if you haven't already
cd src
uvicorn server:app --reload --port 8000
```

Then, in this directory:

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173` (or wherever Vite picks). No env file is needed for
local dev: when `VITE_BACKEND_WS_URL` is unset, `useChordStream.js` falls back to
`ws://${window.location.hostname}:8000/stream`, which matches the backend command
above. Set it in `.env.development` only if your backend lives somewhere else.

Lint and production build:

```bash
npm run lint     # oxlint
npm run build    # vite build -> dist/
```

## Using it

While idle the UI shows a source picker:

- **Microphone** — plain `getUserMedia`.
- **Tab / System Audio** — `getDisplayMedia`, plus a link that opens YouTube in a new
  tab so you can go find something to play.

Then a current-chord display, connection status, start/stop button, and a short
history of recent chord changes.

## How it works

- **`src/useChordStream.js`** — owns one listening session end to end
  (`start(source)` / `stop()`), where `source` is `"mic"` or `"display"`.
- **`public/audio-processor.js`** — an `AudioWorkletProcessor` that buffers raw
  samples into ~4096-sample chunks before handing them to the main thread, so we send
  roughly 12 WebSocket messages/sec instead of ~375 (one per 128-sample render
  callback).
- **`src/App.jsx`** — the UI described above.

Everything downstream of capture is source-agnostic: it only ever sees a `MediaStream`
with an audio track, regardless of where that track came from.

### Tab / system audio specifics

`getDisplayMedia({ video: true, audio: true })` requests video even though we never
use it — Chrome only shows the "share tab audio" checkbox when video is also
requested. The video track is stopped and discarded immediately after the stream
arrives.

If the resulting stream has **zero audio tracks** (the user picked "Entire Screen" or a
window instead of a tab, or left "share audio" unchecked), the whole stream is torn
down and a descriptive error is thrown rather than silently connecting a dead stream.

### Sample rate

The `AudioContext` is created with `sampleRate: 48000` to match what the model was
trained on (`model.RATE` in the backend) — the Web Audio API resamples input to match
automatically, so there's no manual resampling code here. If a browser doesn't honor
that constructor option, `useChordStream.js` logs a console warning
(`audioCtx.sampleRate` won't equal 48000) rather than silently mispredicting.

### Session lifecycle

Three guards worth knowing about, each of which fixed a real bug:

- **`sessionActiveRef`** — a *synchronous* lock checked and set at the very top of
  `start()`, before any `await`. A second `start()` while one is live returns
  immediately as a no-op. It deliberately does **not** tear down and take over from
  the first session: two in-flight async flows sharing the same refs cross-talk badly
  (closing a still-connecting WebSocket wakes the first flow into its own `catch`,
  which then tears down the *second* session). `stop()` releases the lock.
- **`ended` listener on the captured audio track** — unplugging the mic, closing the
  shared tab, or hitting Chrome's own "Stop sharing" bar resets the UI to idle instead
  of leaving it stuck on "Listening" over a dead stream.
- **`useEffect` unmount cleanup** — calls `stop()` on unmount so navigating away
  releases the mic/socket rather than waiting on the browser's own teardown.

Each of these was verified with a throwaway Playwright script during development
(headless Chromium, `--use-fake-device-for-media-stream` for mic and
`--auto-select-desktop-capture-source` for tab audio). Those scripts were never
committed — there is currently **no test suite in this repo**. Worth adding before
touching session lifecycle again; the double-start race in particular is invisible on
a code read and obvious the moment you fire two clicks in one tick.

## Deploying

See [`../FIREBASE_DEPLOY.md`](../FIREBASE_DEPLOY.md) in the repo root — this app
deploys to Firebase Hosting and needs `VITE_BACKEND_WS_URL` set to your deployed
backend's Cloud Run URL at build time (see `.env.production.example`).

The WebSocket connects **directly** to the Cloud Run URL rather than through a Firebase
Hosting rewrite. Cloud Run supports WebSockets natively; WebSocket-through-Hosting-rewrite
behaviour isn't clearly documented, so we don't rely on it.
