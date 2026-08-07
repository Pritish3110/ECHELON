from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Loads the embedding model. 
        Using paraphrase-multilingual-MiniLM-L12-v2 for Hindi/English support (384-dim).
        """
        self.model = SentenceTransformer(model_name)
        
    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()
