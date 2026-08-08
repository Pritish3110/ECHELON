import time
import logging
import numpy as np
import sys
import queue
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice.audio import AudioCapture
from src.voice.vad import VADFilter
from src.voice.asr import ASR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def test_asr_live():
    log.info("Initializing Zipformer ASR (CPU)...")
    
    vad = VADFilter(threshold=0.5)
    asr = ASR()
    
    log.info("Warming up ASR...")
    asr.transcribe(np.zeros(16000, dtype=np.float32))
    
    print("\n" + "="*60)
    print("Live ASR Test (Indian English Zipformer)")
    print("Speak clearly into the microphone. It will transcribe after you stop speaking.")
    print("Press Ctrl+C to exit.")
    print("="*60 + "\n")
    
    capture = AudioCapture()
    
    try:
        while True:
            capture.start()
            print("Listening... (Speak now, stop speaking to process)")
            
            audio_buffer = []
            silence_frames = 0
            is_speaking = False
            
            SILENCE_THRESHOLD_FRAMES = 45 # Matches the 1.35s natural conversation pacing from main.py
            
            while True:
                try:
                    chunk = capture.get_chunk(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check VAD
                if vad.is_speech(chunk):
                    if not is_speaking:
                        is_speaking = True
                        print("Speech detected! Keep speaking...")
                    silence_frames = 0
                    audio_buffer.append(chunk)
                else:
                    if is_speaking:
                        silence_frames += 1
                        audio_buffer.append(chunk)
                        if silence_frames > SILENCE_THRESHOLD_FRAMES:
                            print("Silence threshold reached, processing audio...")
                            break
            
            capture.stop()
            
            if audio_buffer:
                audio_data = np.concatenate(audio_buffer)
                
                # Measure ASR Latency
                t0 = time.time()
                text = asr.transcribe(audio_data)
                asr_latency = time.time() - t0
                
                print(f"\n[ASR Result]: '{text}'")
                print(f"[Latency]: {asr_latency:.2f} seconds\n")
                print("-" * 60)
                
                # Give a short pause before listening again
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nExiting live test.")
        capture.stop()

if __name__ == "__main__":
    test_asr_live()
