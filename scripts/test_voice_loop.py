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
from src.llm.groq import GroqClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def test_pipeline():
    log.info("Initializing models... (Monitor VRAM via 'nvidia-smi' in another terminal)")
    
    # Load all models simultaneously to test concurrent VRAM footprint
    vad = VADFilter(threshold=0.5)
    asr = ASR(model_size="small") # small supports Hindi and English, medium.en is English only
    tts = TTS(lang_code='a')
    llm = GroqClient()
    
    log.info("Warming up models to eliminate first-run latency...")
    asr.transcribe(np.zeros(16000, dtype=np.float32))
    tts.synthesize("warm up")
    
    log.info("All models loaded and warmed up successfully.")
    
    phrases_to_test = 5
    
    print("\n" + "="*60)
    print("Multilingual Latency Test (English & Hindi)")
    print(f"We will test {phrases_to_test} phrases.")
    print("Please monitor nvidia-smi during this test for VRAM fragmentation.")
    print("="*60 + "\n")
    
    capture = AudioCapture()
    
    for i in range(phrases_to_test):
        print(f"\n--- Test {i+1}/{phrases_to_test} ---")
        input("Press ENTER, then start speaking...")
        
        capture.start()
        print("Listening... (stop speaking to finish)")
        
        audio_buffer = []
        silence_frames = 0
        is_speaking = False
        
        # 500ms silence = 500 / 30 = ~16 frames
        SILENCE_THRESHOLD_FRAMES = 16 
        
        try:
            while True:
                try:
                    chunk = capture.get_chunk(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check VAD
                if vad.is_speech(chunk):
                    if not is_speaking:
                        is_speaking = True
                        print("Speech detected!")
                    silence_frames = 0
                    audio_buffer.append(chunk)
                else:
                    if is_speaking:
                        silence_frames += 1
                        audio_buffer.append(chunk) # Keep silence in buffer for natural end
                        if silence_frames > SILENCE_THRESHOLD_FRAMES:
                            print("Silence detected, processing...")
                            break
        except KeyboardInterrupt:
            print("\nInterrupted.")
            capture.stop()
            break
            
        capture.stop()
        
        if not audio_buffer:
            print("No audio captured.")
            continue
            
        audio_data = np.concatenate(audio_buffer)
        
        # Measure ASR Latency
        t0 = time.time()
        text = asr.transcribe(audio_data)
        asr_latency = time.time() - t0
        print(f"[ASR] ({asr_latency:.2f}s): {text}")
        
        if not text:
            print("No text recognized. Skipping LLM.")
            continue
            
        # Measure LLM Latency
        t0 = time.time()
        system_prompt = "You are a helpful voice assistant. Answer concisely, in the same language as the user's prompt (e.g. Hindi or English). Limit response to 1-2 short sentences."
        response = llm.generate(text, system_prompt)
        llm_latency = time.time() - t0
        print(f"[LLM] ({llm_latency:.2f}s via groq): {response}")
        
        # Measure TTS Latency
        t0 = time.time()
        waveform = tts.synthesize(response)
        tts_latency = time.time() - t0
        print(f"[TTS] ({tts_latency:.2f}s): <audio generated>")
        
        total_latency = asr_latency + llm_latency + tts_latency
        print(f"[Total Processing Latency]: {total_latency:.2f}s")
        
        # Play audio
        if waveform is not None and len(waveform) > 0:
            print("Playing response...")
            play_audio(waveform, samplerate=24000)
            
if __name__ == "__main__":
    test_pipeline()
