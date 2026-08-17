# Deploying RTCD: Firebase Hosting + Cloud Run

This is what to do **after** you've created your Firebase project in the console.
Two separate things get deployed:

- **The React frontend** -> Firebase Hosting (static files).
- **The Python backend** (`src/server.py`, wraps `ChordEngine`) -> Cloud Run.
  Firebase itself can't run a PyTorch model directly (Cloud Functions aren't
  built for a persistent WebSocket + a multi-hundred-MB ML dependency stack) --
  Cloud Run is the natural fit since it's a real container, supports
  WebSockets natively, and lives in the same Google Cloud project as your
  Firebase project (a Firebase project *is* a GCP project under the hood).

Both were exercised locally as part of this project: the FastAPI/WebSocket
server was verified end-to-end with a real audio file streamed in over a
WebSocket, and the whole browser pipeline (mic -> AudioWorklet -> WebSocket ->
prediction -> UI) was verified with an automated headless-browser test using a
fake microphone device. What was **not** possible to verify from this sandbox:
actually building the Docker image (container registry pulls are blocked
here) or a live deployment (needs your real GCP/Firebase credentials). Steps
2 and 5 below are where you'll be doing that verification yourself.

## 0. Prerequisites

- Your Firebase project must be on the **Blaze (pay-as-you-go) plan** --
  Cloud Run isn't available on the free Spark plan. You still only pay for
  what you use; Cloud Run's free tier is generous for a personal project.
- Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and the
  [Firebase CLI](https://firebase.google.com/docs/cli) (`npm install -g
  firebase-tools`) if you don't have them.
- `gcloud auth login` and `firebase login` -- both need to point at the same
  Google account that owns the Firebase project.
- `gcloud config set project YOUR-PROJECT-ID` (find this in Firebase console
  -> Project settings -> Project ID, not the display name).

## 1. Enable the APIs you need

```
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

Cloud Build is what actually builds the Docker image in the cloud in step 2
-- you don't need Docker installed locally at all.

## 2. Deploy the backend to Cloud Run

From the **repo root** (where the `Dockerfile` lives):

```
cur
```

Notes on the flags:
- `--source .` tells gcloud to build the container via Cloud Build from your
  local `Dockerfile` and push it -- no local `docker build` needed.
- `--allow-unauthenticated` lets the browser connect without a Google auth
  token. Fine for a personal project; if you add Firebase Auth later (see
  the "Auth" option you mentioned) and want to gate access, this is the flag
  you'd revisit.
- `--min-instances=1` keeps one instance warm at all times. **This costs
  money continuously** (roughly a small fraction of a cent per hour for a
  small instance, but it's not free) -- the reason to set it is that Cloud
  Run's default scale-to-zero means the *first* request after idle has to
  cold-start a container that imports torch/librosa and loads the model
  checkpoint, which is not a fast operation and would make your first chord
  prediction after any idle period noticeably delayed. If cost matters more
  than that latency for you right now, drop this flag (or set
  `--min-instances=0`) and accept a slow first connection after idle periods.
- `--memory=2Gi` -- torch + librosa + matplotlib is a heavier stack than
  Cloud Run's 512Mi default handles comfortably. Bump further if you see
  out-of-memory errors in the Cloud Run logs.

This will take a few minutes (building the image, pushing it, deploying).
When it finishes, it prints a **Service URL** like
`https://chordthingy-backend-xxxxxxxxxx-uc.a.run.app` -- copy it, you need it
next.

**Verify it before moving on:**
```
curl https://YOUR-SERVICE-URL/
```
should return `{"status":"ok","device":"cpu","rate":48000}`. If this doesn't
work, the frontend won't either -- check `gcloud run services logs read
chordthingy-backend` for the actual error before continuing.

## 3. Point the frontend at the deployed backend

```
cd frontend
cp .env.production.example .env.production
```

Edit `.env.production` and set:
```
VITE_BACKEND_WS_URL=wss://YOUR-SERVICE-URL/stream
```
(same host as step 2's Service URL, but `wss://` instead of `https://`, and
`/stream` instead of `/`.)

This value gets baked into the built JS at build time -- if you ever
redeploy the backend to a different URL, you need to update this and rebuild.

**Why the frontend talks directly to the Cloud Run URL instead of going
through Firebase Hosting:** Firebase Hosting can rewrite plain HTTP requests
to a Cloud Run service, but WebSocket proxying through that same rewrite
layer has historically been inconsistent/unclear across Firebase's docs and
versions. Connecting the browser straight to the Cloud Run service's own
URL for the WebSocket sidesteps that ambiguity entirely -- Cloud Run
natively supports WebSockets with no caveats. Firebase Hosting is still
doing real work here, just for the static frontend files, not for proxying
the socket.

## 4. Build and deploy the frontend

```
npm run buildfire
cd ..
firebase use --add
```
(`firebase use --add` will prompt you to pick your project from a list and
create `.firebaserc` -- this repo doesn't ship one since it's specific to
your project.)

```
firebase deploy --only hosting
```

This uploads `frontend/dist` (per `firebase.json`) to Firebase Hosting and
prints your live URL, something like `https://YOUR-PROJECT.web.app`.

## 5. Test it for real

Open the printed Hosting URL in a browser, click "Start Listening", allow
mic access, and play some chords. Open the browser console if anything looks
wrong -- connection errors there usually mean either the Cloud Run URL in
`.env.production` is wrong, or the backend isn't actually up (recheck the
curl health check from step 2).

## What's NOT set up yet (by design, per your earlier answers)

- **Auth / Firestore history**: you asked to keep Firebase minimal for the
  first pass beyond hosting+backend. Adding Firebase Auth (gate the mic
  behind login) or Firestore (save session history) are both additive later
  -- neither requires re-architecting what's here.
- **File-upload mode**: you chose mic-only first. The backend's `/`
  endpoint and WebSocket `/stream` are the only routes; a future
  `POST /analyze-file` endpoint (already sketched in the project roadmap)
  would need its own Firebase Hosting rewrite if you want it same-origin, or
  can also just be called directly against the Cloud Run URL like the
  WebSocket is.
- **Tightening CORS**: `src/server.py` currently allows all origins
  (`allow_origins=["*"]`). Once you have a stable Hosting URL, consider
  narrowing that to just your Hosting domain.
