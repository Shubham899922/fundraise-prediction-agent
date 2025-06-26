from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_nlp_features(pitch: str):
    """
    Extract semantic embedding from the pitch using a transformer model.
    """
    return model.encode(pitch)

def extract_tabular_features(meta: dict):
    """
    Extract numeric features from project metadata.
    """
    return np.array([
        meta.get("team_size", 0),
        meta.get("twitter_followers", 0),
        meta.get("past_exits", 0),
        meta.get("has_prototype", 0)
    ])
