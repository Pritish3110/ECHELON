import os
import io
import wave
import numpy as np
import logging
import sherpa_onnx
from groq import Groq
from src.config.settings import settings

log = logging.getLogger(__name__)

class CloudASR:
    def __init__(self):
        self.api_key = settings.groq_api_key
        if not self.api_key or self.api_key == "mock_key":
            self.client = None
            log.warning("Groq API key not found. Cloud ASR disabled.")
        else:
            self.client = Groq(api_key=self.api_key)

    def transcribe(self, audio: np.ndarray) -> str:
        if not self.client:
            raise Exception("Cloud ASR disabled.")
            
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((audio * 32767.0).astype(np.int16).tobytes())
        
        wav_io.seek(0)
        
        transcription = self.client.audio.transcriptions.create(
            file=('audio.wav', wav_io.read()),
            model='whisper-large-v3-turbo',
            prompt="ECHELON, terminal, browser, close, open, brightness, volume, hi, hello, system.",
            temperature=0.0
        )
        
        # Whisper hallucinates "Thank you." or "you" on silence occasionally. 
        text = transcription.text.strip()
        if text.lower() in ["thank you.", "thank you", "you.", "you"]:
            return ""
            
        return text

class LocalASR:
    def __init__(self):
        log.info(f"Loading Zipformer (sherpa-onnx) Indian English model on CPU (Fallback)...")
        
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "zipformer"
        )
        
        encoder_path = os.path.join(model_dir, "encoder-epoch-10-avg-5-chunk-64-left-256.int8.onnx")
        decoder_path = os.path.join(model_dir, "decoder-epoch-10-avg-5-chunk-64-left-256.int8.onnx")
        joiner_path = os.path.join(model_dir, "joiner-epoch-10-avg-5-chunk-64-left-256.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        
        bpe_path = os.path.join(model_dir, "bpe.vocab")
        hotwords_path = os.path.join(model_dir, "hotwords.txt")
        
        if not os.path.exists(tokens_path):
            log.error(f"Zipformer model not found at {model_dir}. Please run scripts/download_zipformer.py")
        
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=False,
            decoding_method="modified_beam_search",
            max_active_paths=4,
            modeling_unit="bpe",
            bpe_vocab=bpe_path,
            hotwords_file=hotwords_path,
            hotwords_score=2.0,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        audio = audio.flatten().astype(np.float32)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio.tolist())
        
        tail_padding = np.zeros(int(0.5 * 16000), dtype=np.float32)
        stream.accept_waveform(16000, tail_padding.tolist())
        stream.input_finished()
        
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
            
        return self.recognizer.get_result(stream).strip()

class ASR:
    def __init__(self, model_size="small", compute_type="int8"):
        """
        Hybrid ASR strategy:
        1. Try Groq's whisper-large-v3-turbo API (blazing fast, flawless accent handling).
        2. Fallback to local Zipformer CPU model if network fails.
        """
        self.cloud_asr = CloudASR()
        self.local_asr = LocalASR()

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            log.debug("Attempting cloud transcription via Groq...")
            return self.cloud_asr.transcribe(audio)
        except Exception as e:
            log.warning(f"Cloud ASR failed ({e}). Falling back to local Zipformer...")
            return self.local_asr.transcribe(audio)
