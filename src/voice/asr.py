from faster_whisper import WhisperModel
import numpy as np
import logging

log = logging.getLogger(__name__)

class ASR:
    def __init__(self, model_size="small", compute_type="int8"):
        """
        Using 'small' instead of 'medium.en' because the user requested
        multilingual testing (Hindi and English). 
        """
        log.info(f"Loading Faster-Whisper {model_size} ({compute_type})...")
        self.model = WhisperModel(model_size, device="cuda", compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        # Faster-whisper expects 1D float32 array
        audio = audio.flatten()
        
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,               # low latency
            condition_on_previous_text=False, # avoid streaming drift
            initial_prompt="Hello, Echelon. Could you please help me?"
        )
        
        text = "".join(segment.text for segment in segments)
        return text.strip()
