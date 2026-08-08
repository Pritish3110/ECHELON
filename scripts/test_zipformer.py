import os
import wave
import sherpa_onnx
import numpy as np

def create_recognizer():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "zipformer")
    
    encoder_path = os.path.join(model_dir, "encoder-epoch-10-avg-5-chunk-64-left-256.int8.onnx")
    decoder_path = os.path.join(model_dir, "decoder-epoch-10-avg-5-chunk-64-left-256.int8.onnx")
    joiner_path = os.path.join(model_dir, "joiner-epoch-10-avg-5-chunk-64-left-256.int8.onnx")
    tokens_path = os.path.join(model_dir, "tokens.txt")
    
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
        max_active_paths=4,
    )
    return recognizer

if __name__ == "__main__":
    recognizer = create_recognizer()
    stream = recognizer.create_stream()
    
    # create a dummy audio array
    audio = np.zeros(16000, dtype=np.float32)
    stream.accept_waveform(16000, audio.tolist())
    
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
        
    print(f"Recognized text: {recognizer.get_result(stream)}")
    print("Zipformer loaded successfully!")
