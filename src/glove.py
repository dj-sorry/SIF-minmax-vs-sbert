import os
import requests
import zipfile
import numpy as np

def download_glove(output_dir):
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    zip_file = 'glove.6B.zip'

    if not os.path.exists(os.path.join(output_dir, zip_file)):
        print(f"Downloading {zip_file}...")
        response = requests.get(url)
        with open(os.path.join(output_dir, zip_file), 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"{zip_file} already exists.")

    if not os.path.exists(os.path.join(output_dir, 'glove.6B')):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(os.path.join(output_dir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete.")
    else:
        print("Extracted files already exist.")

    return os.path.join(output_dir, 'glove.6B', 'glove.6B.300d.txt')

def load_glove_vectors(glove_file_path):

    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def calculate_glove_corpus_embeddings(corpus, glove_embeddings):
    embeddings = []
    embedding_dim = len(next(iter(glove_embeddings.values())))
    
    for texts in corpus:
        text_embeddings = []
        for text in texts:
            words = text.split()
            text_embedding = np.zeros(embedding_dim)
            valid_words = 0
            
            for word in words:
                if word in glove_embeddings:
                    text_embedding += glove_embeddings[word]
                    valid_words += 1
            
            if valid_words > 0:
                text_embedding /= valid_words 
            
            text_embeddings.append(text_embedding)
        
        embeddings.append(text_embeddings)
    
    return embeddings
