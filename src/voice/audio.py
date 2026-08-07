import queue
import sounddevice as sd
import numpy as np

class AudioCapture:
    def __init__(self, samplerate=16000, channels=1, dtype='float32'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.q = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}", flush=True)
        self.q.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                     dtype=self.dtype, callback=self._callback, blocksize=512) # 512 samples (~32ms) for Silero VAD
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_chunk(self, block=True, timeout=None):
        return self.q.get(block=block, timeout=timeout)

def play_audio(waveform: np.ndarray, samplerate: int = 24000):
    """Play the synthesized waveform."""
    sd.play(waveform, samplerate)
    sd.wait()
