"""
Survey preparation: building comparison pairs, generating HTML tables
for evaluation, parsing survey results, and computing Bradley-Terry scores.
"""

from __future__ import annotations

import re
import random
import itertools

import markdown
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Comparison-pair construction
# ---------------------------------------------------------------------------

def build_comparison_pairs(
    type_items: list[tuple[str, list[tuple[str, str]]]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """
    Build all pairwise comparison pairs from categorised outline items.

    *type_items* is a list of ``(type_name, items)`` where each *items*
    entry is ``(text, label)``.  Every combination of two distinct types
    is paired, and within each type-pair every item combination is
    generated with random left/right ordering.

    Returns a list of ``((textA, labelA), (textB, labelB))`` tuples.
    """
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for (_, items_a), (_, items_b) in itertools.combinations(type_items, 2):
        for item_a, item_b in itertools.product(items_a, items_b):
            pair = [item_a, item_b]
            random.shuffle(pair)
            pairs.append(tuple(pair))  # type: ignore[arg-type]
    return pairs


# ---------------------------------------------------------------------------
# HTML table generation
# ---------------------------------------------------------------------------

def _split_main_claim_and_reasons(text: str) -> tuple[str, str]:
    """Split text into main-claim and reasons parts."""
    text = text.strip()
    match = re.search(r"\*?\*?Reason\s*(\(1\))?:\*?\*?", text)
    if match:
        return text[: match.start()].strip(), text[match.start() :].strip()
    return text, ""


def _md_to_html(md_text: str) -> str:
    return markdown.markdown(md_text, extensions=["extra"])


def generate_html_tables(
    comparison_pairs: list[tuple[tuple[str, str], tuple[str, str]]],
) -> str:
    """
    Render comparison pairs as side-by-side HTML tables suitable for
    embedding in a survey tool (e.g. Qualtrics).
    """
    tables: list[str] = []
    for idx, pair in enumerate(comparison_pairs):
        (text_a, label_a), (text_b, label_b) = pair
        a_main, a_reasons = _split_main_claim_and_reasons(text_a)
        b_main, b_reasons = _split_main_claim_and_reasons(text_b)

        table = f"""
<!-- Comparison Pair {idx + 1}: {label_a[0] + label_a[-1]} vs {label_b[0] + label_b[-1]} -->
<table border="1" cellpadding="1" cellspacing="1" style="width:800px;">
  <tbody>
    <tr>
      <td style="text-align: left;"><span style="font-size:16px;">
      {_md_to_html(a_main)}</span></td>
      <td style="text-align: left;"><span style="font-size:16px;">
      {_md_to_html(b_main)}</span></td>
    </tr>
    <tr>
      <td style="text-align: left;"><span style="font-size:16px;">
      {_md_to_html(a_reasons)}</span></td>
      <td style="text-align: left;"><span style="font-size:16px;">
      {_md_to_html(b_reasons)}</span></td>
    </tr>
  </tbody>
</table>
"""
        tables.append(table)
    return "\n\n".join(tables)


# ---------------------------------------------------------------------------
# Survey-result parsing
# ---------------------------------------------------------------------------

_HEADER_PATTERN = re.compile(r"^(r|k)([noce]\d+)([noce]\d+)#(\d+)_(\d+)$")


def parse_survey_results(rows: list[list[str]]) -> dict[str, list[tuple[str, str]]]:
    """
    Parse raw survey-sheet rows into match results.

    *rows* is expected to have:
    - row 0: column headers (e.g. ``rsgn1o1#1_1``)
    - row 1: left-outline win counts
    - row 2: right-outline win counts

    Returns a dict mapping ``question_<N>`` to a list of
    ``(winner, loser)`` tuples.
    """
    headers = rows[0]
    left_outcomes = rows[1]
    right_outcomes = rows[2]

    results: dict[str, list[tuple[str, str]]] = {}

    for col, left_win, right_win in zip(
        headers[1:], left_outcomes[1:], right_outcomes[1:]
    ):
        m = _HEADER_PATTERN.match(col)
        if not m:
            continue

        _group, player1, player2, _bracket, match_number = m.groups()

        if left_win > right_win:
            winner, loser = player1, player2
        elif right_win > left_win:
            winner, loser = player2, player1
        else:
            print("no winner")
            continue

        results.setdefault(f"question_{match_number}", []).append((winner, loser))

    return results


def merge_match_results(
    *result_dicts: dict[str, list[tuple[str, str]]],
) -> dict[str, list[tuple[str, str]]]:
    """Merge multiple match-result dicts into one."""
    merged: dict[str, list[tuple[str, str]]] = {}
    for d in result_dicts:
        for key, matches in d.items():
            merged.setdefault(key, []).extend(matches)
    return merged


# ---------------------------------------------------------------------------
# Bradley-Terry model
# ---------------------------------------------------------------------------

PLAYERS = ["n1", "n2", "o1", "o2", "c1", "c2", "e1", "e2"]


def compute_bradley_terry(
    matches: list[tuple[str, str]],
    players: list[str] | None = None,
) -> dict[str, float]:
    """
    Fit a Bradley-Terry model to a list of ``(winner, loser)`` pairs
    and return a dict mapping each player label to its estimated score.
    """
    if players is None:
        players = PLAYERS

    n_players = len(players)
    player_to_idx = {p: i for i, p in enumerate(players)}
    initial = np.ones(n_players)

    def neg_log_likelihood(scores: np.ndarray) -> float:
        s = np.exp(scores)
        ll = 0.0
        for winner, loser in matches:
            i, j = player_to_idx[winner], player_to_idx[loser]
            ll += np.log(s[i] / (s[i] + s[j]))
        return -ll

    result = minimize(neg_log_likelihood, initial, method="BFGS")
    scores = np.exp(result.x)

    score_dict: dict[str, float] = {}
    for player, score in zip(players, scores):
        score_dict[player] = float(score)
        print(f"Player {player}: {score}")

    return score_dict
