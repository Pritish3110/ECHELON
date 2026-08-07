import time
import logging
import numpy as np
import sys
import queue
import os

# Auto-inject LD_LIBRARY_PATH for CUDA 12 (required by faster-whisper/ctranslate2)
venv_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
cublas_dir = os.path.join(venv_lib_dir, "nvidia", "cublas", "lib")
cudnn_dir = os.path.join(venv_lib_dir, "nvidia", "cudnn", "lib")

current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if cublas_dir not in current_ld and os.path.exists(cublas_dir):
    new_ld = f"{cublas_dir}:{cudnn_dir}"
    if current_ld:
        new_ld += f":{current_ld}"
    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice.audio import AudioCapture, play_audio
from src.voice.vad import VADFilter
from src.voice.asr import ASR
from src.voice.tts import TTS
from src.llm.fallback import FallbackLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_echelon():
    log.info("Initializing ECHELON models...")
    
    vad = VADFilter(threshold=0.5)
    asr = ASR(model_size="small") 
    tts = TTS(lang_code='a')
    llm = FallbackLLM()
    
    log.info("Models loaded. System ready.")
    
    capture = AudioCapture()
    SILENCE_THRESHOLD_FRAMES = 16 # ~500ms at 30ms chunks
    
    try:
        while True:
            print("\n" + "="*40)
            print("Waiting for speech... (Ctrl+C to exit)")
            capture.start()
            
            audio_buffer = []
            silence_frames = 0
            is_speaking = False
            
            while True:
                try:
                    chunk = capture.get_chunk(timeout=0.1)
                except queue.Empty:
                    continue
                
                if vad.is_speech(chunk):
                    if not is_speaking:
                        is_speaking = True
                        print("Listening...")
                    silence_frames = 0
                    audio_buffer.append(chunk)
                else:
                    if is_speaking:
                        silence_frames += 1
                        audio_buffer.append(chunk)
                        if silence_frames > SILENCE_THRESHOLD_FRAMES:
                            break
            
            capture.stop()
            
            if not audio_buffer:
                continue
                
            audio_data = np.concatenate(audio_buffer)
            
            # ASR
            print("Processing speech...")
            text = asr.transcribe(audio_data)
            print(f"You said: {text}")
            
            if not text:
                continue
                
            # LLM
            response = llm.generate(text, "Keep answers brief and conversational.")
            print(f"ECHELON: {response}")
            
            # TTS
            waveform = tts.synthesize(response)
            if waveform is not None and len(waveform) > 0:
                play_audio(waveform, samplerate=24000)
                
    except KeyboardInterrupt:
        capture.stop()
        print("\nExiting ECHELON.")

if __name__ == "__main__":
    run_echelon()
