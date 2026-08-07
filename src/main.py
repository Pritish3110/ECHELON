import time
import logging
import numpy as np
import sys
import queue
import os
import warnings

warnings.filterwarnings("ignore")

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
from src.llm.groq import GroqLLM
from src.agent.router import IntentRouter
from src.tools.file_ops.handler import FileOpsHandler
from src.tools.system_control.handler import SystemControlHandler
from src.memory.buffer import ConversationBuffer
from src.tools.rag_query.handler import RAGHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("phonemizer").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

def run_echelon():
    log.info("Initializing ECHELON models...")
    
    vad = VADFilter(threshold=0.5)
    asr = ASR(model_size="small") 
    tts = TTS(lang_code='a')
    llm = GroqLLM()
    router = IntentRouter()
    
    log.info("Models loaded. System ready.")
    
    capture = AudioCapture()
    SILENCE_THRESHOLD_FRAMES = 16 # ~500ms at 30ms chunks
    
    def ask_callback(prompt_msg: str) -> bool:
        """Callback to pause execution and await user confirmation via voice."""
        print(f"ECHELON SECURITY: {prompt_msg}")
        clean_prompt = prompt_msg.replace("/", " slash ").replace("_", " underscore ").replace(".", " dot ")
        waveform = tts.synthesize(clean_prompt)
        if waveform is not None and len(waveform) > 0:
            play_audio(waveform, samplerate=24000)
            
        print("Listening for confirmation (say 'yes' or 'no')...")
        capture.start()
        start_t = time.time()
        is_spk = False
        silence = 0
        buf = []
        while True:
            if not is_spk and (time.time() - start_t) > 6.0:
                break
            try:
                ch = capture.get_chunk(timeout=0.1)
            except queue.Empty:
                continue
            if vad.is_speech(ch):
                if not is_spk: is_spk = True
                silence = 0
                buf.append(ch)
            elif is_spk:
                silence += 1
                buf.append(ch)
                if silence > 16:
                    break
        capture.stop()
        
        if not buf:
            print("No confirmation heard. Denying by default.")
            return False
            
        ans = asr.transcribe(np.concatenate(buf, axis=0)).strip().lower()
        print(f"Heard confirmation: '{ans}'")
        return any(w in ans for w in ["yes", "yep", "sure", "ok", "yeah", "do it"])
        
    file_handler = FileOpsHandler(ask_callback=ask_callback)
    system_handler = SystemControlHandler()
    rag_handler = RAGHandler()
    memory_buffer = ConversationBuffer()
    
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
                
            # Route Intent
            print("Routing intent...")
            route = router.classify_intent(text)
            print(f"Intent classified as: {route}")
            
            # Synthesize Response
            context = memory_buffer.get_context_string(turns=2)
            
            if route == "general_chat":
                response = llm.generate(text, "Keep answers brief and conversational.", memory_buffer.get_full_history())
            elif route == "file_ops":
                print("Executing FileOps Handler...")
                response = file_handler.handle(text, context)
            elif route == "system_control":
                print("Executing SystemControl Handler...")
                response = system_handler.handle(text, context)
            elif route == "rag_query":
                print("Executing RAG Query Handler...")
                response = rag_handler.handle(text, context)
            elif route == "memory_recall":
                print("Executing Memory Recall...")
                sys_prompt = "You are ECHELON. The user is asking about something you discussed recently. Use the conversation history to answer concisely."
                response = llm.generate(text, sys_prompt, memory_buffer.get_full_history())
            else:
                response = f"Routing to {route.replace('_', ' ')}. Tool execution is coming in the next phase."
                
            print(f"ECHELON: {response}")
            memory_buffer.add_turn(text, response)
            
            # TTS
            waveform = tts.synthesize(response)
            if waveform is not None and len(waveform) > 0:
                play_audio(waveform, samplerate=24000)
                
    except KeyboardInterrupt:
        capture.stop()
        print("\nExiting ECHELON.")

if __name__ == "__main__":
    run_echelon()
