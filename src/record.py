import pyaudio
import wave
import threading
import queue


class Recording:
    def __init__(self, CHUNK=1024, FORMAT=pyaudio.paInt16, CHANNELS=1, RATE=48000):
        self.CHUNK = CHUNK
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.RATE = RATE

        #frames will hold the audio data in raw byte format
        #ex. frames[0] = b'\x00\x00\x00\x00\x00\x00\x00\x00...' (1 chunk of audio data)
        #    frames[1] = b'\x00\x00\x00\x00\x00\x00\x00\x00...'
        #    ...
        self.frames = []
        self.audio_queue = queue.Queue()
        self.recording = False
        self.thread = None

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            rate=self.RATE,
            channels=self.CHANNELS,
            format=self.FORMAT,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def start_recording(self):
        stream = self.stream
        data = stream.read(self.CHUNK)
        self.frames.append(data)
        return data
    
    def start_stream(self):
        self.recording = True
        self.thread = threading.Thread(target=self._record_loop) #start a new thread continuously get audio and put into queue
        self.thread.start()

    def _record_loop(self):
        while self.recording:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)
            self.audio_queue.put(data)
        
    def stop_stream(self):
        self.recording = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def save_recording(self, filename="output.wav"):
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b"".join(self.frames))
        wf.close()

    def __del__(self):
        self.stop_stream()
        self.p.terminate()

