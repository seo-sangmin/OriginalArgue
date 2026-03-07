"""
Plotting similarity trends and running Mann-Kendall trend tests.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
from matplotlib.ticker import MaxNLocator


def plot_similarity_curves(
    similarities_dict: dict[str, list[float]],
    y_label: str = "Semantic Textual Similarity",
    y_limits: tuple[float, float] | None = None,
    caption: str = "",
) -> None:
    """
    Plot one or more similarity curves and display Mann-Kendall test
    results as a table underneath.

    *similarities_dict* maps a human-readable label to a list of
    similarity values (one per enhancement step).

    Style conventions:
    - "Non-Enhanced" / "Irrelevant" lines are dotted baselines.
    - Labels containing "Simply" use triangle markers and dashed lines.
    - Labels containing "Originality" are red; "Cogency" labels are green.
    """
    STYLE_MAP = {
        "Non-Enhanced":                          dict(marker=",", linestyle=":", linewidth=1.5),
        "Originality Enhanced Simply":           dict(marker="^", linestyle="--", color="red", linewidth=2),
        "Originality Enhanced":                  dict(marker="o", linestyle="--", color="red", linewidth=2),
        "Cogency and Originality Enhanced Simply": dict(marker="^", linestyle="-", color="green", linewidth=2),
        "Cogency and Originality Enhanced":      dict(marker="o", linestyle="-", color="green", linewidth=2),
        "Irrelevant":                            dict(marker=",", linestyle=":", linewidth=1.5),
    }

    plt.figure(figsize=(12, 8))

    for label, values in similarities_dict.items():
        style = STYLE_MAP.get(label, {})
        plt.plot(values, label=label, **style)

    if y_limits:
        plt.ylim(*y_limits)

    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=12))
    plt.gca().yaxis.set_minor_locator(MaxNLocator(nbins=24))
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.xlabel("Number of Enhancement", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(fontsize=14, loc="lower right", frameon=True, title_fontsize=10)
    plt.grid(visible=True, linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Mann-Kendall test table ------------------------------------------
    # Only test the "enhanced" curves (skip baselines)
    enhanced = {
        k: v for k, v in similarities_dict.items()
        if k not in ("Non-Enhanced", "Irrelevant")
    }
    if not enhanced:
        return

    results = {
        "Method": list(enhanced.keys()),
        "Trend":   [mk.original_test(v).trend for v in enhanced.values()],
        "p-value": [mk.original_test(v).p     for v in enhanced.values()],
        "z-score": [mk.original_test(v).z     for v in enhanced.values()],
        "Slope":   [mk.original_test(v).slope for v in enhanced.values()],
    }
    df = pd.DataFrame(results)
    print(df.to_latex(index=False, float_format="%.4f"))
    styled = df.style.set_caption(caption or "Mann-Kendall Trend Test Results")
    # Display in notebook environments
    try:
        from IPython.display import display
        display(styled)
    except ImportError:
        print(df.to_string(index=False))
