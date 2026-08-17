import { useCallback, useEffect, useRef, useState } from "react";

const SAMPLE_RATE = 48000;

//set env to gcloud run backend url, otherwise default to localhost:8000
const BACKEND_WS_URL =
  import.meta.env.VITE_BACKEND_WS_URL || `ws://${window.location.hostname}:8000/stream`;

const HISTORY_LIMIT = 20;

export function useChordStream() {
  const [status, setStatus] = useState("idle"); // idle | connecting | listening | error
  const [currentChord, setCurrentChord] = useState(null);
  const [history, setHistory] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);

  const audioCtxRef = useRef(null);
  const streamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const wsRef = useRef(null);
  const sessionActiveRef = useRef(false);



  //main cleanup funciton
  const stop = useCallback(() => {
    sessionActiveRef.current = false;

    workletNodeRef.current?.port.close();
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close();
    }
    audioCtxRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;

    setStatus((s) => (s === "error" ? s : "idle"));
  }, []);

  const start = useCallback(async (source = "mic") => {

    //if a session is already active, don't start a new one
    if (sessionActiveRef.current) {
      return;
    }
    sessionActiveRef.current = true;

    //ui 
    setErrorMessage(null);
    setStatus("connecting");
    setHistory([]);
    setCurrentChord(null);


    try {
      const ws = new WebSocket(BACKEND_WS_URL);
      wsRef.current = ws;

      //recieve backend messages and update ui
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setCurrentChord(data.chord);
        setHistory((prev) => [...prev.slice(-(HISTORY_LIMIT - 1)), data]);
      };
      ws.onclose = () => {
        setStatus((s) => (s === "error" ? s : "idle"));
      };

      await new Promise((resolve, reject) => {
        ws.addEventListener("open", resolve, { once: true });
        ws.addEventListener(
          "error",
          () => reject(new Error("Could not reach the chord detection server.")),
          { once: true },
        );
      });

      let stream; //stream assigned to either mic or display capture
      if (source === "display") { //use audio tab input -------------------------------
        //tab audio capture
        const displayStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true,
        });

        const audioTracks = displayStream.getAudioTracks();
        if (audioTracks.length === 0) { //error if no audio track(tab) is shared
          displayStream.getTracks().forEach((track) => track.stop());
          throw new Error(
            'No audio was shared. Reopen and pick a browser tab (not "Entire Screen" or ' +
              '"Window"), and make sure "Share tab audio" is checked.',
          );
        }

        displayStream.getVideoTracks().forEach((track) => track.stop());
        stream = new MediaStream(audioTracks);
      } else { //use mic input ---------------------------------------------------------
        //channel 1 = mono
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1 },
        });
      }
      streamRef.current = stream;

      
      stream.getAudioTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          if (sessionActiveRef.current) stop();
        });
      });

      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = audioCtx;

      if (audioCtx.sampleRate !== SAMPLE_RATE) {
        console.warn(
          `AudioContext ignored the requested ${SAMPLE_RATE}Hz sample rate ` +
            `(got ${audioCtx.sampleRate}Hz instead) -- predictions may be ` +
            `inaccurate, since the model was trained on ${SAMPLE_RATE}Hz audio.`,
        );
      }

      await audioCtx.audioWorklet.addModule("/audio-processor.js"); //audio processer to accumulate audio chunks and send to backend via websocket
      const mediaNode = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, "chunker-processor");
      workletNodeRef.current = workletNode;

      //worklet will send audio chunks roughly every 128 samples, since its too small, the chunker processor in audio-processor.js will accumulate them into 4096 sample chunks and send to backend via websocket
      workletNode.port.onmessage = (event) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(event.data.buffer);
        }
      };
      mediaNode.connect(workletNode);

      setStatus("listening");
    } catch (err) {
      console.error(err);
      setErrorMessage(err.message || String(err));
      setStatus("error");
      stop();
    }
  }, [stop]);

  //stop on unmount
  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { status, currentChord, history, errorMessage, start, stop };
}
