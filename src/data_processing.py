"""
Loading, filtering, and searching existing argument texts from the
ChickWard/ConnEli dataset on Hugging Face.
"""

from __future__ import annotations

import re
import random
import statistics

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_argument_texts() -> list[str]:
    """
    Load the ChickWard/ConnEli dataset and return a flat list of the
    ``output`` field from every example.
    """
    dataset = load_dataset("ChickWard/ConnEli", split="train")
    texts = [example["output"] for example in dataset]
    print(f"Loaded {len(texts)} argument texts.")
    return texts


# ---------------------------------------------------------------------------
# Filtering by word count
# ---------------------------------------------------------------------------

def filter_by_word_count(
    texts: list[str],
    low: int = 200,
    high: int = 3200,
) -> list[str]:
    """
    Remove texts whose word count falls below *low* or above *high*.
    Returns the filtered list.
    """
    indices_to_remove = {
        i
        for i, text in enumerate(texts)
        if len(text.split()) < low or len(text.split()) > high
    }
    print("Indices removed:", sorted(indices_to_remove))

    filtered = [t for i, t in enumerate(texts) if i not in indices_to_remove]
    print(f"Remaining texts: {len(filtered)}")
    return filtered


# ---------------------------------------------------------------------------
# Word-count statistics
# ---------------------------------------------------------------------------

def print_word_count_stats(texts: list[str]) -> None:
    """Print median and mean word counts for a list of texts."""
    counts = [len(t.split()) for t in texts]
    print(f"Median word count: {statistics.median(counts)}")
    print(f"Mean word count:   {statistics.mean(counts):.2f}")


# ---------------------------------------------------------------------------
# Relevance search
# ---------------------------------------------------------------------------

# Patterns related to propositional modularity / functional discreteness
RELEVANCE_PATTERNS = [
    r"propositional modularity",
    r"propositionally modular",
    r"functional discreteness",
    r"functionally discrete",
    r"discrete causal",
    r"discrete and causal",
    r"discreteness and causal",
    r"discretely causal",
    r"discretely active",
    r"causal discrete",
    r"causal and discrete",
    r"causality and discrete",
    r"causally discrete",
]


def find_relevant_indices(
    texts: list[str],
    patterns: list[str] | None = None,
) -> list[int]:
    """
    Return sorted indices of *texts* that match any of the given regex
    *patterns* (case-insensitive).  Defaults to ``RELEVANCE_PATTERNS``.
    """
    if patterns is None:
        patterns = RELEVANCE_PATTERNS

    matched: set[int] = set()
    for pattern in patterns:
        for idx, text in enumerate(texts):
            if re.search(pattern, text, re.IGNORECASE):
                matched.add(idx)
                print(f"Found '{pattern}' at index {idx}")

    sorted_indices = sorted(matched)
    print(f"Total relevant texts found: {len(sorted_indices)}")
    return sorted_indices


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def sample_indices(total: int, num_samples: int) -> list[int]:
    """
    Return *num_samples* random indices from ``range(total)``.
    If *num_samples* >= *total*, return all indices in order.
    """
    if num_samples < total:
        return random.sample(range(total), num_samples)
    return list(range(total))
