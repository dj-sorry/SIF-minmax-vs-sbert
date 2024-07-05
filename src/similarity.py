import numpy as np
from typing import List

def min_max_scale(similarities: List[float]) -> List[float]:
    min_val = np.min(similarities)
    max_val = np.max(similarities)
    return [(sim - min_val) / (max_val - min_val) for sim in similarities]

def calculate_similarity(embeddings1: List[np.ndarray], embeddings2: List[np.ndarray]) -> List[float]:
    similarities = []
    for i in range(len(embeddings1)):
        sim = np.dot(embeddings1[i], embeddings2[i])
        similarities.append(sim)
    return min_max_scale(similarities)
