import { useState } from "react";
import { useChordStream } from "./useChordStream";
import "./App.css";

function App() {
  const { status, currentChord, history, errorMessage, start, stop } = useChordStream();
  const [source, setSource] = useState("mic");

  const isListening = status === "listening" || status === "connecting";
  // Only let the source be changed while nothing is running -- switching it
  // mid-session wouldn't do anything until the next Start anyway, and hiding
  // the picker avoids implying otherwise.
  const canChangeSource = !isListening;

  return (
    <div className="app">
      <h1>RTCD</h1>
      <p className="subtitle">Real-Time Chord Detection</p>

      {canChangeSource && (
        <div className="source-select">
          <button
            type="button"
            className={`source-option ${source === "mic" ? "source-option--active" : ""}`}
            onClick={() => setSource("mic")}
          >
            Microphone
          </button>
          <button
            type="button"
            className={`source-option ${source === "display" ? "source-option--active" : ""}`}
            onClick={() => setSource("display")}
          >
            Tab / System Audio
          </button>
        </div>
      )}

      {canChangeSource && source === "display" && (
        <p className="source-hint">
          For a YouTube song: open{" "}
          <a href="https://www.youtube.com" target="_blank" rel="noreferrer">
            YouTube
          </a>{" "}
          in another tab, start playing the song, then come back here and press Start -- pick
          that tab in the picker and check &quot;Share tab audio&quot;.
        </p>
      )}

      <div className="chord-display">
        <span className={`chord-name ${currentChord ? "" : "chord-name--empty"}`}>
          {currentChord ?? "—"}
        </span>
      </div>

      <button
        type="button"
        className={`toggle-button ${isListening ? "toggle-button--stop" : ""}`}
        onClick={isListening ? stop : () => start(source)}
      >
        {status === "connecting" ? "Connecting..." : isListening ? "Stop" : "Start Listening"}
      </button>

      <p className={`status status--${status}`}>
        {status === "idle" && "Not listening"}
        {status === "connecting" &&
          (source === "display"
            ? "Connecting to server and waiting for you to pick a tab to share..."
            : "Connecting to server and requesting mic access...")}
        {status === "listening" && "Listening"}
        {status === "error" && (errorMessage || "Something went wrong")}
      </p>

      {history.length > 0 && (
        <div className="history">
          <h2>Recent chords</h2>
          <ul>
            {history
              .slice()
              .reverse()
              .map((entry, i) => (
                <li key={`${entry.time}-${i}`}>
                  <span className="history-time">{entry.time.toFixed(2)}s</span>
                  <span className="history-chord">{entry.chord}</span>
                </li>
              ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
