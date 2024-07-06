import numpy as np
from typing import List

def min_max_scale(similarities: List[float]) -> List[float]:
    """
    Scales a list of similarity scores to the range [0, 1] using minmax.
    
    Parameters:
    similarities (List[float]): A list of similarity scores.
    
    Returns:
    List[float]: A list of similarity scores scaled to the range [0, 1].
    """
    min_val = np.min(similarities)
    max_val = np.max(similarities)
    return [(sim - min_val) / (max_val - min_val) for sim in similarities]

def calculate_similarity(embeddings1: List[np.ndarray], embeddings2: List[np.ndarray]) -> List[float]:
    """
    Calculates the similarity between two lists of embeddings and scales the results using Min-Max scaling.
    
    Parameters:
    embeddings1 (List[np.ndarray]): A list of embeddings (numpy arrays).
    embeddings2 (List[np.ndarray]): A list of embeddings (numpy arrays) of the same length as embeddings1.
    
    Returns:
    List[float]: A list of similarity scores scaled to the range [0, 1].
    """
    similarities = []
    for i in range(len(embeddings1)):
        sim = np.dot(embeddings1[i], embeddings2[i])
        similarities.append(sim)
    return min_max_scale(similarities)
