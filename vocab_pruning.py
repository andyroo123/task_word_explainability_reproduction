# vocab_pruning.py
import numpy as np
from typing import Tuple, Optional

def compute_word_features(
    image_embeddings: np.ndarray,
    word_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute word-level scalar features for each image by taking
    the dot product between image embeddings and word embeddings.

    Returns:
        features: np.ndarray of shape (N, V)
            features[n, j] = <image_n, word_j>
    """
    return image_embeddings @ word_embeddings.T


def score_words_by_correlation(
    features: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Compute an absolute Pearson correlation score between each
    word feature column and the binary label vector y.

    Returns:
        scores: np.ndarray of shape (V,), higher = more task-relevant
    """
    y = y.astype(float)
    y_centered = y - y.mean()
    denom_y = np.sqrt((y_centered ** 2).sum()) + 1e-8

    N, V = features.shape
    scores = np.zeros(V, dtype=float)

    feats_centered = features - features.mean(axis=0, keepdims=True)
    denom_feats = np.sqrt((feats_centered ** 2).sum(axis=0)) + 1e-8

    numerators = (feats_centered * y_centered[:, None]).sum(axis=0)

    corr = numerators / (denom_feats * denom_y)
    scores = np.abs(corr)

    return scores


def prune_vocabulary(
    image_embeddings: np.ndarray,
    word_embeddings: np.ndarray,
    y: np.ndarray,
    top_k: Optional[int] = None,
    top_percent: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prune the word vocabulary based on supervised relevance scores.

    Exactly one of top_k or top_percent should be provided.

    Args:
        image_embeddings: (N, D) CLIP image embeddings.
        word_embeddings: (V, D) CLIP text embeddings for all words.
        y: (N,) binary labels (0/1).
        top_k: keep the top_k most relevant words.
        top_percent: keep the top_percent fraction of words (e.g., 0.1 for top 10%).

    Returns:
        pruned_word_embeddings: (V_pruned, D)
        pruned_indices: (V_pruned,) indices into original vocab
        scores: (V,) full score array (before pruning)
    """
    assert (top_k is None) != (top_percent is None), \

    features = compute_word_features(image_embeddings, word_embeddings)

    scores = score_words_by_correlation(features, y)

    V = word_embeddings.shape[0]
    if top_percent is not None:
        top_k = max(1, int(V * top_percent))

    pruned_indices = np.argsort(scores)[::-1][:top_k]
    pruned_word_embeddings = word_embeddings[pruned_indices]

    return pruned_word_embeddings, pruned_indices, scores
