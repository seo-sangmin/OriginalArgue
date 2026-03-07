# OriginalArgue

Generating cogent and original argument outlines through iterative critique-and-revision with large language models (GPT-4o, o1, o1-preview). This project accompanies the master's thesis by Sangmin Seo.

## Project Structure

```
OriginalArgue/
├── main.py                  # Pipeline runner with section toggle flags
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Prompt templates, claims, and constants
│   ├── api_client.py        # OpenAI Chat Completions API wrappers
│   ├── data_io.py           # Save/upload/download via Hugging Face Hub
│   ├── data_processing.py   # Dataset loading, filtering, relevance search
│   ├── embeddings.py        # Vertex AI text embedding & cosine similarity
│   ├── visualization.py     # Similarity trend plots & Mann-Kendall tests
│   ├── survey.py            # Comparison pairs, HTML tables, Bradley-Terry model
│   └── finetuning.py        # Fine-tuning dataset preparation & validation
└── Generating Cogent and Original Arguments v3.6.ipynb  # Original notebook
```

## Pipeline Overview

1. **Outline Generation** — Generate normal, originality-enhanced, and cogency+originality-enhanced argument outlines via iterative critique-and-revision loops.
2. **Existing Texts** — Load the ChickWard/ConnEli dataset, revise texts, synthesise outlines, and identify relevant arguments by keyword patterns.
3. **Embeddings & Similarity** — Embed outlines with Vertex AI (`text-embedding-005`), compute cosine similarities against reference outlines, and plot trends.
4. **Survey & Bradley-Terry** — Build pairwise comparison tables for human evaluation and fit a Bradley-Terry model to the survey results.
5. **Fine-Tuning** — Prepare OpenAI fine-tuning datasets (simple and instruction-tuned), validate format, and estimate token costs.

## Setup

```bash
pip install -r requirements.txt
```

Set the following environment variables before running:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API access |
| `HF_ACCESS_TOKEN` | Hugging Face Hub uploads |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI embeddings (or authenticate via `gcloud`) |

## Usage

```bash
python main.py
```

Toggle individual pipeline sections by editing the flags at the top of `main.py`:

```python
GPT_GEN = False                # True = call APIs; False = use cached data
RUN_OUTLINE_GENERATION = True
RUN_EXISTING_TEXTS = True
RUN_EMBEDDINGS = True
RUN_SURVEY = True
RUN_FINETUNING = True
```

For Google Colab, uncomment the authentication block at the top of `main.py`.

## Module Reference

| Module | Responsibility |
|---|---|
| `config.py` | All prompt templates, evaluation criteria, claim definitions, background texts, and shared constants (seeds, iteration counts) |
| `api_client.py` | `generate_responses()` for single-shot generation; `improve_outlines()` for iterative critique → revision loops |
| `data_io.py` | JSONL serialisation, Hugging Face upload/download, Markdown export, response extraction from process logs |
| `data_processing.py` | Load ChickWard/ConnEli, filter by word count, regex-based relevance search across argument texts |
| `embeddings.py` | Vertex AI embedding (single & batched), cosine similarity, average similarity against reference sets |
| `visualization.py` | Matplotlib line plots with Mann-Kendall trend test tables (LaTeX & styled DataFrame output) |
| `survey.py` | Combinatorial pair generation, HTML table rendering for survey tools, survey-result parsing, Bradley-Terry optimisation |
| `finetuning.py` | Build chat-format datasets, OpenAI format validation, token counting via tiktoken, epoch & cost estimation |
