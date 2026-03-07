"""
Preparing, validating, and estimating costs for OpenAI fine-tuning datasets.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import tiktoken


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_simple_finetune_dataset(
    system_instruction: str,
    outlines: list[str],
    revised_texts: list[str],
) -> list[dict]:
    """
    Build a fine-tuning dataset where every example uses the same
    *system_instruction*.

    Returns a list of ``{"messages": [...]}`` dicts ready for JSONL export.
    """
    return [
        {
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": outline},
                {"role": "assistant", "content": text},
            ]
        }
        for outline, text in zip(outlines, revised_texts)
    ]


def build_instruction_finetune_dataset(
    per_example_instructions: list[str],
    outlines: list[str],
    revised_texts: list[str],
) -> list[dict]:
    """
    Build a fine-tuning dataset where each example has its own synthesised
    system instruction.
    """
    return [
        {
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": outline},
                {"role": "assistant", "content": text},
            ]
        }
        for instruction, outline, text in zip(
            per_example_instructions, outlines, revised_texts
        )
    ]


def save_finetune_dataset(dataset: list[dict], path: str = "gpt_finetune_dataset.jsonl") -> None:
    """Write the dataset to a JSONL file."""
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(dataset)} examples to {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_finetune_dataset(dataset: list[dict]) -> dict[str, int]:
    """
    Run format-error checks on a fine-tuning dataset, mirroring the
    checks recommended by OpenAI.

    Returns a dict of error-type counts (empty if no errors).
    """
    errors: dict[str, int] = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            errors["data_type"] += 1
            continue

        messages = ex.get("messages")
        if not messages:
            errors["missing_messages_list"] += 1
            continue

        for msg in messages:
            if "role" not in msg or "content" not in msg:
                errors["message_missing_key"] += 1
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in msg):
                errors["message_unrecognized_key"] += 1
            if msg.get("role") not in ("system", "user", "assistant", "function"):
                errors["unrecognized_role"] += 1

            content = msg.get("content")
            fn_call = msg.get("function_call")
            if (not content and not fn_call) or not isinstance(content, str):
                errors["missing_content"] += 1

        if not any(m.get("role") == "assistant" for m in messages):
            errors["example_missing_assistant_message"] += 1

    if errors:
        print("Found errors:")
        for k, v in errors.items():
            print(f"  {k}: {v}")
    else:
        print("No errors found")

    return dict(errors)


# ---------------------------------------------------------------------------
# Token counting & cost estimation
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_message_tokens(
    messages: list[dict],
    tokens_per_message: int = 3,
    tokens_per_name: int = 1,
) -> int:
    """Approximate token count for a list of chat messages."""
    total = 0
    for msg in messages:
        total += tokens_per_message
        for key, value in msg.items():
            total += len(_ENCODING.encode(value))
            if key == "name":
                total += tokens_per_name
    total += 3  # reply priming
    return total


def count_assistant_tokens(messages: list[dict]) -> int:
    """Count tokens in assistant messages only."""
    return sum(
        len(_ENCODING.encode(m["content"]))
        for m in messages
        if m["role"] == "assistant"
    )


def print_distribution(values: list[int | float], name: str) -> None:
    """Print summary statistics for a list of values."""
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def estimate_training_cost(
    dataset: list[dict],
    max_tokens_per_example: int = 16385,
    target_epochs: int = 3,
    min_target_examples: int = 100,
    max_target_examples: int = 25000,
    min_default_epochs: int = 1,
    max_default_epochs: int = 25,
) -> None:
    """
    Print dataset statistics and an estimated training-token cost,
    following OpenAI's default epoch heuristics.
    """
    n_missing_system = 0
    n_missing_user = 0
    n_messages_list: list[int] = []
    convo_lens: list[int] = []
    assistant_lens: list[int] = []

    for ex in dataset:
        msgs = ex["messages"]
        if not any(m["role"] == "system" for m in msgs):
            n_missing_system += 1
        if not any(m["role"] == "user" for m in msgs):
            n_missing_user += 1
        n_messages_list.append(len(msgs))
        convo_lens.append(count_message_tokens(msgs))
        assistant_lens.append(count_assistant_tokens(msgs))

    print(f"Num examples missing system message: {n_missing_system}")
    print(f"Num examples missing user message:   {n_missing_user}")
    print_distribution(n_messages_list, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_lens, "num_assistant_tokens_per_example")

    n_too_long = sum(1 for length in convo_lens if length > 128_000)
    print(f"\n{n_too_long} examples may be over the 128 000 token limit, "
          "they will be truncated during fine-tuning")

    # Epoch estimation
    n_train = len(dataset)
    n_epochs = target_epochs
    if n_train * target_epochs < min_target_examples:
        n_epochs = min(max_default_epochs, min_target_examples // n_train)
    elif n_train * target_epochs > max_target_examples:
        n_epochs = max(min_default_epochs, max_target_examples // n_train)

    billing_tokens = sum(min(max_tokens_per_example, l) for l in convo_lens)
    print(f"\nDataset has ~{billing_tokens} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * billing_tokens} tokens")
