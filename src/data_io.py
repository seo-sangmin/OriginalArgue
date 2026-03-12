"""
Utilities for saving, uploading, downloading, and post-processing
generation data via Hugging Face Hub and local JSONL/Markdown files.
"""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download, upload_file

from .config import CLAIM_RSG, CLAIM_CLARK


# ---------------------------------------------------------------------------
# Save & upload
# ---------------------------------------------------------------------------

def save_and_upload(
    data: list,
    filename: str,
    repo_name: str,
    access_token: str,
) -> None:
    """Write *data* to a local JSONL file and upload it to Hugging Face."""
    jsonl_path = f"{filename}.jsonl"
    with open(jsonl_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo=jsonl_path,
        repo_id=repo_name,
        repo_type="dataset",
        token=access_token,
    )


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def save_process_markdown(process_data: list, filename: str) -> None:
    """
    Write generation processes to a Markdown file.

    *process_data* may be either a flat list of process strings or a nested
    list (list of lists).
    """
    md_path = f"{filename}.md"
    try:
        with open(md_path, "w") as md_file:
            if process_data and isinstance(process_data[0], list):
                for list_idx, process_list in enumerate(process_data):
                    md_file.write(f"\n\n# List {list_idx + 1}\n\n")
                    for proc_idx, process in enumerate(process_list):
                        md_file.write(f"## Process {proc_idx + 1}\n\n")
                        md_file.write(f"{process.strip()}\n\n")
            else:
                for proc_idx, process in enumerate(process_data):
                    md_file.write(f"## Process {proc_idx + 1}\n\n")
                    md_file.write(f"{process.strip()}\n\n")
    except Exception as e:
        print(f"An error occurred while saving to {md_path}: {e}")


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------

def extract_responses_from_processes(
    process_listlist: list[list[str]],
    remove_critiques: bool,
) -> list[list[str]]:
    """
    Given a nested list of process strings, extract the ``### Response:``
    portion from each entry.

    When *remove_critiques* is True every second response (the critique) is
    dropped.  Each response is prepended with the appropriate claim based on
    its position in the outer list.
    """
    response_listlist = []
    total = len(process_listlist)
    for idx, process_list in enumerate(process_listlist):
        responses = [
            proc.split("### Response:")[-1].strip() for proc in process_list
        ]
        if remove_critiques:
            responses = responses[0::2]
        claim = CLAIM_RSG if idx < total // 2 else CLAIM_CLARK
        response_listlist.append([claim + r for r in responses])
    return response_listlist


def extract_flat_responses(process_list: list[str]) -> list[str]:
    """Extract ``### Response:`` content from a flat list of processes."""
    return [proc.split("### Response:")[-1].strip() for proc in process_list]


# ---------------------------------------------------------------------------
# Download & process (combined helper)
# ---------------------------------------------------------------------------

def download_and_process(
    repo_name: str,
    filename: str,
    remove_critiques: bool,
) -> tuple[list, list]:
    """
    Download a JSONL file from Hugging Face, write a Markdown copy, and
    return ``(process_data, response_data)``.

    The type of *response_data* depends on whether the downloaded data is a
    nested list (returns list-of-lists via ``extract_responses_from_processes``)
    or a flat list (returns a flat list via ``extract_flat_responses``).
    """
    filepath = hf_hub_download(
        repo_id=repo_name,
        filename=f"{filename}.jsonl",
        repo_type="dataset",
    )

    process_data: list = []
    with open(filepath, "r") as f:
        for line in f:
            process_data.append(json.loads(line))

    save_process_markdown(process_data, filename)

    if process_data and isinstance(process_data[0], list):
        response_data = extract_responses_from_processes(
            process_data, remove_critiques
        )
    else:
        response_data = extract_flat_responses(process_data)

    return process_data, response_data


# ---------------------------------------------------------------------------
# Download raw embeddings
# ---------------------------------------------------------------------------

def download_embeddings(repo_name: str, filename: str) -> list:
    """Download a JSONL file of embeddings and return them as a list."""
    filepath = hf_hub_download(
        repo_id=repo_name,
        filename=f"{filename}.jsonl",
        repo_type="dataset",
    )
    embeddings: list = []
    with open(filepath, "r") as f:
        for line in f:
            embeddings.append(json.loads(line))
    return embeddings
