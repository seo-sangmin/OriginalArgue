"""
Text embedding via Vertex AI and cosine-similarity computation.
"""

from __future__ import annotations

import time

import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def embed_text(
    texts: list[str],
    dimensionality: int = 768,
    task: str = "SEMANTIC_SIMILARITY",
) -> list[list[float]]:
    """
    Embed a list of texts using Vertex AI ``text-embedding-005``.

    Returns a list of embedding vectors, one per input text.
    """
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]


def embed_in_batches(
    texts: list[str],
    batch_size: int = 12,
    sleep_time: int = 10,
) -> list[list[float]]:
    """
    Embed *texts* in batches, sleeping between batches to stay within
    API rate limits.  Returns a flat list of embedding vectors.
    """
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(embed_text(batch))
        print(f"Embedded texts {start}:{start + batch_size}")
        time.sleep(sleep_time)
    return all_embeddings


def embed_nested_lists(
    response_listlist: list[list[str]],
    batch_size: int = 12,
    sleep_time: int = 10,
) -> list[list[list[float]]]:
    """
    Embed each inner list separately and return a nested list of
    embedding vectors matching the structure of *response_listlist*.
    """
    return [
        embed_in_batches(responses, batch_size, sleep_time)
        for responses in response_listlist
    ]


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def average_similarity(
    embeddings_a: list[list[float]],
    embeddings_b: list[list[float]],
) -> list[float]:
    """
    For each embedding in *embeddings_a*, compute its average cosine
    similarity against all embeddings in *embeddings_b*.

    Returns a list of average similarity scores (one per embedding in A).
    """
    num_b = len(embeddings_b)
    return [
        sum(cosine_similarity(a, b) for b in embeddings_b) / num_b
        for a in embeddings_a
    ]


def similarities_against_reference(
    outline_embeddings: list[list[list[float]]],
    reference_embeddings,
) -> list[float]:
    """
    Compute the average similarity between each list of outline embeddings
    and a set of reference embeddings.

    *reference_embeddings* may be either:
    - a flat list of embedding vectors, or
    - a pair ``[ref_rsg, ref_clark]`` when the first half of
      *outline_embeddings* should be compared to ``ref_rsg`` and the
      second half to ``ref_clark``.

    Returns a single list of per-step average similarities (averaged
    across all inner lists at each position).
    """
    has_split_refs = (
        isinstance(reference_embeddings[0][0], list)
    )
    half = len(outline_embeddings) // 2

    per_list_sims: list[list[float]] = []
    for i, embs in enumerate(outline_embeddings):
        if has_split_refs:
            ref = reference_embeddings[0] if i < half else reference_embeddings[1]
        else:
            ref = reference_embeddings
        per_list_sims.append(average_similarity(embs, ref))

    print(per_list_sims)

    # Average across lists at each position
    avg = [sum(vals) / len(vals) for vals in zip(*per_list_sims)]
    print(avg)
    return avg
