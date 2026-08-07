from kokoro import KPipeline
import numpy as np
import logging

log = logging.getLogger(__name__)

class TTS:
    def __init__(self, lang_code='a', voice='af_heart'):
        log.info(f"Loading Kokoro-82M TTS (lang={lang_code}, voice={voice})...")
        self.pipeline = KPipeline(lang_code=lang_code) # 'a' = American English
        self.voice = voice

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.array([])
            
        generator = self.pipeline(
            text, voice=self.voice, speed=1.0, split_pattern=r'\n+'
        )
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                audio_chunks.append(audio)
                
        if not audio_chunks:
            return np.array([])
            
        return np.concatenate(audio_chunks)
