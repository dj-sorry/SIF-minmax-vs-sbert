# %%
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets import load_dataset
from scipy import stats
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pandas as pd

from sif import compute_word_frequencies, compute_sif_weights, compute_sif_embeddings, remove_pc_sif
from similarity import calculate_similarity
from glove import load_glove_vectors
from prepare_data import preprocess_text

# %%
ds = load_dataset("mteb/stsbenchmark-sts")
glove_vectors = load_glove_vectors("../data/raw/glove.6B.300d.txt")

# %%
df_sts = ds["train"].data.to_pandas()
df_sts['sentence1'] = preprocess_text(df_sts['sentence1'])
df_sts['sentence2'] = preprocess_text(df_sts['sentence2'])
sentences = df_sts[['sentence1', 'sentence2']].values.flatten().tolist()

# %%
word_freq = compute_word_frequencies(sentences)
sif_weights = compute_sif_weights(word_freq)
corpus = [df_sts['sentence1'].tolist(), df_sts['sentence2'].tolist()]

# %%
#compute SIF

embeddings1 = compute_sif_embeddings(corpus[0], glove_vectors, sif_weights)
embeddings2 = compute_sif_embeddings(corpus[1], glove_vectors, sif_weights)
embeddings1_pc_removed = remove_pc_sif(embeddings1)
embeddings2_pc_removed = remove_pc_sif(embeddings2)
similarities = calculate_similarity(embeddings1_pc_removed, embeddings2_pc_removed)
scaled_similarities = minmax_scale(similarities)

df_sts['minmax_similarity'] = scaled_similarities

# %%
#compute SBERT

model = SentenceTransformer('all-MiniLM-L6-v2')
df_sts['sentence1'] = df_sts['sentence1'].apply(' '.join)
df_sts['sentence2'] = df_sts['sentence2'].apply(' '.join)
embeddings1 = model.encode(df_sts['sentence1'].tolist())
embeddings2 = model.encode(df_sts['sentence2'].tolist())

sbert_similarities = [cosine_similarity(embeddings1[i].reshape(1, -1), 
                                        embeddings2[i].reshape(1, -1))[0][0] 
                      for i in range(len(df_sts))]

sbert_correlation = np.corrcoef(df_sts['score'], sbert_similarities)[0, 1]
df_sts['sbert_similarity'] = sbert_similarities


# %%
#compute BM-25

tokenized_corpus1 = preprocess_text(df_sts['sentence1'])
tokenized_corpus2 = preprocess_text(df_sts['sentence2'])

tokenized_corpus_combined = pd.concat([tokenized_corpus1, tokenized_corpus2])
bm25 = BM25Okapi(tokenized_corpus_combined.tolist())

bm25_scores = []
for i in range(len(df_sts)):
    query = tokenized_corpus2[i]
    scores = bm25.get_scores(query)
    bm25_scores.append(scores[i])

df_sts['bm25_similarity'] = minmax_scale(bm25_scores)


# %%
df_sts

# %%
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
sns.histplot(df_sts['score'], bins=20, kde=True)
plt.title('Distribution of Actual Scores')

plt.subplot(1, 4, 2)
sns.histplot(df_sts['minmax_similarity'], bins=20, kde=True)
plt.title('Distribution of SIF-minmax Similarities')

plt.subplot(1, 4, 3)
sns.histplot(df_sts['sbert_similarity'], bins=20, kde=True)
plt.title('Distribution of SBERT Similarities')

plt.tight_layout()
plt.show()


# %%
def analyze_errors(df_sts):
    df_sts['score_normalized'] = df_sts['score'] / 5

    df_sts['error_minmax'] = df_sts['score_normalized'] - df_sts['minmax_similarity']
    df_sts['error_sbert'] = df_sts['score_normalized'] - df_sts['sbert_similarity']

    plt.figure(figsize=(10, 6))

    sns.histplot(df_sts['error_minmax'], bins=20, kde=True, color='blue', label='SIF-minmax')
    sns.histplot(df_sts['error_sbert'], bins=20, kde=True, color='red', label='SBERT', alpha=0.5)

    plt.title('Error Distribution for SIF-minmax and SBERT Similarities')
    plt.xlabel('Error (Normalized Score - Similarity)')
    plt.xlim(-1, 1) 
    plt.legend()
    plt.tight_layout()
    plt.show()

analyze_errors(df_sts)


# %%
def compute_statistics(df_sts):

    sbert_pearson_corr = np.corrcoef(df_sts['score'], df_sts['sbert_similarity'])[0, 1]
    sif_pearson_corr = np.corrcoef(df_sts['score'], df_sts['minmax_similarity'])[0, 1]
    bm25_pearson_corr = np.corrcoef(df_sts['score'], df_sts['bm25_similarity'])[0, 1]

    sbert_spearman_corr, _ = stats.spearmanr(df_sts['score'], df_sts['sbert_similarity'])
    sif_spearman_corr, _ = stats.spearmanr(df_sts['score'], df_sts['minmax_similarity'])
    bm25_spearman_corr, _ = stats.spearmanr(df_sts['score'], df_sts['bm25_similarity'])

    sbert_kendall_tau, _ = stats.kendalltau(df_sts['score'], df_sts['sbert_similarity'])
    sif_kendall_tau, _ = stats.kendalltau(df_sts['score'], df_sts['minmax_similarity'])
    bm25_kendall_tau, _ = stats.kendalltau(df_sts['score'], df_sts['bm25_similarity'])

    sbert_mae = mean_absolute_error(df_sts['score_normalized'], df_sts['sbert_similarity'])
    sif_mae = mean_absolute_error(df_sts['score_normalized'], df_sts['minmax_similarity'])
    bm25_mae = mean_absolute_error(df_sts['score_normalized'], df_sts['bm25_similarity'])

    sbert_mse = mean_squared_error(df_sts['score_normalized'], df_sts['sbert_similarity'])
    sif_mse = mean_squared_error(df_sts['score_normalized'], df_sts['minmax_similarity'])
    bm25_mse = mean_squared_error(df_sts['score_normalized'], df_sts['bm25_similarity'])

    sbert_rmse = np.sqrt(sbert_mse)
    sif_rmse = np.sqrt(sif_mse)
    bm25_rmse = np.sqrt(bm25_mse)

    return {
        "sbert": {
            "pearson": sbert_pearson_corr,
            "spearman": sbert_spearman_corr,
            "kendall": sbert_kendall_tau,
            "mae": sbert_mae,
            "mse": sbert_mse,
            "rmse": sbert_rmse,
        },
        "sif": {
            "pearson": sif_pearson_corr,
            "spearman": sif_spearman_corr,
            "kendall": sif_kendall_tau,
            "mae": sif_mae,
            "mse": sif_mse,
            "rmse": sif_rmse,
        },
        "bm25": {
            "pearson": bm25_pearson_corr,
            "spearman": bm25_spearman_corr,
            "kendall": bm25_kendall_tau,
            "mae": bm25_mae,
            "mse": bm25_mse,
            "rmse": bm25_rmse,
        }
    }

statistics = compute_statistics(df_sts)

def visualize_statistics(statistics):
    metrics = ["pearson", "spearman", "kendall", "mae", "mse", "rmse"]
    sbert_values = [statistics["sbert"][metric] for metric in metrics]
    sif_values = [statistics["sif"][metric] for metric in metrics]
    bm25_values = [statistics["bm25"][metric] for metric in metrics]

    df_stats = pd.DataFrame({
        "Metric": metrics,
        "SBERT": sbert_values,
        "SIF": sif_values,
        "BM25": bm25_values
    })

    df_stats_melted = df_stats.melt(id_vars="Metric", var_name="Method", value_name="Value")

    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", 3)  # Modern color palette

    barplot = sns.barplot(data=df_stats_melted, x="Metric", y="Value", hue="Method", palette=palette)
    
    # Adding value labels
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points')

    plt.title('Comparison of Statistical Measures for SBERT, SIF, and BM25', fontsize=16, weight='bold')
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Method', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.show()

visualize_statistics(statistics)



