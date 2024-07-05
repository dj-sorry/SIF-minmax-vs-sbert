from typing import Iterable, List, Dict, Generator
from typing_extensions import deprecated
import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def compute_word_frequencies(sentences: List[List[str]]) -> Dict[str, int]:
    """
    Compute word frequencies which is necessary for calulating SIF weights (word_freq in notebook)
    Use defaultdict to avoid KeyError.
    
    Parameters:
    sentences (List[List[str]]): A nested list where each sublist contains sentences.
    
    Returns:
    Dict[str, int]: A dictionary with words as keys and their frequencies as values.
    """
    word_freq = defaultdict(int)
    for sentence in sentences:
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = sentence
        for word in words:
            word_freq[word] += 1
    return word_freq



def compute_sif_weights(
        word_freq: Dict[str, int], 
        a: float = 1e-3) -> Dict[str, float]:
    """
    Compute Smooth Inverse Frequency (SIF) weights for words.
    Recall that SIF weights are calculated with a/a+p(w) where p(w) is frequency.
    
    Parameters:
    word_freq (Dict[str, int]): A dictionary with words as keys and their frequencies as values.
    a (float): A smoothing parameter, maybe .003, maybe .0003
    
    Returns:
    Dict[str, float]: A dictionary with words as keys and their SIF weights as values.
    """
    return {word: a / (a + freq) for word, freq in word_freq.items()}

"""
@deprecated
def compute_sif_embeddings_queries(corpus: List[List[str]], 
                                   word_vectors: Dict[str, np.ndarray], 
                                   sif_weights: Dict[str, float]) -> List[np.ndarray]:
"""
"""
    Compute SIF-weighted sentence embeddings for a list of queries. This function is customized 
    for the dataset format of queries. The different function are needed at this time because there 
    are several passages (of different lengths: from 6 to 10) for each query.
    
    Parameters:
    corpus (List[List[str]]): A nested list where each sublist contains sentences.
    word_vectors (Dict[str, np.ndarray]): A dictionary with words as keys and their vector representations as values.
    sif_weights (Dict[str, float]): A dictionary with words as keys and their SIF weights as values.
    
    Returns:
    List[np.ndarray]: A list of embeddings for each sentence in the corpus.
    """
"""
    embeddings = []
    for sublist in corpus:
        for sentence in sublist:
            words = sentence.split()
            vectors = []
            weights = []
            for word in words:
                if word in word_vectors and word in sif_weights:
                    vectors.append(word_vectors[word])
                    weights.append(sif_weights[word])
            if vectors:
                vectors = np.array(vectors)
                weights = np.array(weights)
                weighted_avg = np.average(vectors, axis=0, weights=weights)
                embeddings.append(weighted_avg)
            else:
                embeddings.append(np.zeros(next(iter(word_vectors.values())).shape))
    return embeddings

@deprecated
def compute_sif_embeddings_texts(corpus: List[List[str]], 
                                 word_vectors: Dict[str, np.ndarray], 
                                 sif_weights: Dict[str, float]) -> List[List[np.ndarray]]:
    """
"""

        This one is the version for the texts. Notive that an additional empty list is initialized.
        
        Parameters:
        corpus (List[List[str]]): A nested list where each sublist contains texts.
        word_vectors (Dict[str, np.ndarray]): A dictionary with words as keys and their vector representations as values.
        sif_weights (Dict[str, float]): A dictionary with words as keys and their SIF weights as values.
        
        Returns:
        List[List[np.ndarray]]: A list containing embeddings for each sublist of texts.
        """
"""

    embeddings = []
    for texts in corpus:
        text_embeddings = [] #TODO: create one function to account for both formats.
        for text in texts:
            words = text.split()
            vectors = []
            weights = []
            for word in words:
                if word in word_vectors and word in sif_weights:
                    vectors.append(word_vectors[word])
                    weights.append(sif_weights[word])
            if vectors:
                vectors = np.array(vectors)
                weights = np.array(weights)
                weighted_avg = np.average(vectors, axis=0, weights=weights)
                text_embeddings.append(weighted_avg)
            else:
                text_embeddings.append(np.zeros(next(iter(word_vectors.values())).shape))
        embeddings.append(text_embeddings)
    return embeddings
"""
def compute_sif_embeddings(corpus: List[List[str]], word_vectors: Dict[str, np.ndarray], sif_weights: Dict[str, float]) -> List[np.ndarray]:
    embeddings = []
    for sublist in corpus:
        vectors = []
        weights = []
        for sentence in sublist:
            words = sentence.split()
            for word in words:
                if word in word_vectors and word in sif_weights:
                    vectors.append(word_vectors[word])
                    weights.append(sif_weights[word])
        if vectors:
            vectors = np.array(vectors)
            weights = np.array(weights)
            weighted_avg = np.average(vectors, axis=0, weights=weights)
            embeddings.append(weighted_avg)
        else:
            embeddings.append(np.zeros(next(iter(word_vectors.values())).shape))
    return embeddings

def remove_pc_sif(embeddings: List[np.ndarray], 
                  n: int = 1, 
                  alpha: float = 0.0001) -> List[np.ndarray]:
    """
    Remove the first n principal components from each embedding using the SIF method.
    Recall that removing is along these lines: "vs ← vs - u * u^T * vs"
    In our code: 
    #####
    embedding -> vs
    embedding_proj -> u^T * vs
    alpha + embedding_proj -> u * u^T * vs 
    embedding_pc_removed -> "vs - u * u^T * vs" -> this is the value returned. 
    ####
    TODO: docstrings here
    """
    combined_embeddings = np.vstack(embeddings) #gather all embeddings in one matrix...

    svd = TruncatedSVD(n_components=n, n_iter=7, random_state=0) #prin compを取得する
    svd.fit(combined_embeddings)

    embeddings_pc_removed = []
    for embedding in embeddings:
        embedding_proj = np.dot(embedding, svd.components_.T)
        embedding_pc_removed = embedding - alpha * embedding_proj
        embeddings_pc_removed.append(embedding_pc_removed)
    #you dont need to normalize it. The Rust implementation 
    #and the original code do not perform normalization after the pc removal. 
    #However, our implementation can benefit from it because we calulate similarity later.
    embeddings_pc_removed = [normalize(embedding_pc_removed.reshape(1, -1)).flatten() for embedding_pc_removed in embeddings_pc_removed]

    return embeddings_pc_removed



"""
@deprecated
def flatten(lst: Iterable) -> Generator:
    
    Flatten a nested list structure. This is needed to transform the shape of the MS MARCO dataset
    where columns have slightly different formats.
    
    Parameters:
    lst (Iterable): A potentially nested list of elements.
    
    Yields:
    elements of the nested list in a flattened structure.
    
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(item)
        else:
            yield item

"""