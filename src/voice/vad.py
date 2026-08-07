import torch
import numpy as np
import os

class VADFilter:
    def __init__(self, threshold=0.5, sampling_rate=16000):
        # Use locally downloaded silero-vad to bypass GitHub API rate limit
        repo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.cache', 'silero-vad')
        self.model, utils = torch.hub.load(repo_or_dir=repo_dir,
                                           source='local',
                                           model='silero_vad',
                                           force_reload=False,
                                           trust_repo=True)
        self.threshold = threshold
        self.sampling_rate = sampling_rate

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_chunk).float().squeeze()
        # Ensure it's not empty
        if audio_tensor.numel() == 0:
            return False
            
        # Silero VAD expects chunks of at least 512 samples
        if audio_tensor.dim() == 0:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.shape[0] < 512:
            import torch.nn.functional as F
            audio_tensor = F.pad(audio_tensor, (0, 512 - audio_tensor.shape[0]))
            
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()
        return speech_prob > self.threshold
