import os
from huggingface_hub import snapshot_download

def download_model():
    print("Downloading Zipformer Indian English model...")
    model_id = "Akshatkasera007/STT-streaming-zipformer-indian-en"
    local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "zipformer")
    
    os.makedirs(local_dir, exist_ok=True)
    
    # We only need the onnx files and tokens.txt, but it's safe to just download the snapshot
    # Usually the files are encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print(f"Model downloaded to {local_dir}")

if __name__ == "__main__":
    download_model()
