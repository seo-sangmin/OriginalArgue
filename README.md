\****The codes in this repo are being modularized and corrected based on the original notebook written in Jan 2025.***

# OriginalArgue

Generating cogent and original argument outlines through iterative critique-and-revision with large language models (GPT-4o, o1-preview, o1). This project accompanies the master's thesis by Sangmin Seo.

## Research Summary

This repository implements the experiments from the master's thesis *Leveraging Large Language Models to Generate Original and Cogent Arguments on the Connectionist Eliminativism of Belief* (Sangmin Seo, Munich Center for Mathematical Philosophy, LMU Munich, February 2025). The full text is in [`original_notebook/`](original_notebook/Master_s_Thesis__Sangmin_SEO__v2_8.pdf).

### Main Claim

**Research question.** Can LLMs generate original and cogent philosophical arguments based on specific claims in the debate surrounding the *connectionist eliminativism of belief*?

**Hypothesis.** Properly configured LLMs, combined with methods that enhance originality and cogency, can generate arguments on connectionist eliminativism that are significantly more original and cogent than those produced by simply prompting a single LLM.

Following the thesis, an argument is **cogent** when its reasons are *plausible* (likely to be true) and genuinely *support* the claim, and **original** when its reasons are not the same as those of existing arguments.

The study fixes two opposing argument claims (the "main claims" given to the model as input) drawn from the debate:

1. *Propositional modularity is incompatible with connectionist models of cognitive systems.* — the Ramsey–Stich–Garon / Stich position (`CLAIM_RSG`).
2. *Functionally discrete beliefs, playing a causal role, can be compatible with connectionist models of cognitive systems.* — the Clark position (`CLAIM_CLARK`).

Two further claims — "although there are no propositionally modular beliefs, eliminativism regarding belief is not justified" and Searle's Chinese Room claim — serve as *irrelevant* lower baselines (`CLAIM_SMOLSTICH`, `CLAIM_CHROOM`).

### Methods

1. **Outline enhancement via critique-and-revision loops.** Each argument outline is generated, then iteratively critiqued and revised for 10 iterations. Two prompt variants are compared — *originality-only* and *cogency + originality* — at two depths: *simple* (each revision sees only the latest outline and its critique) and *full* (each revision sees the entire history of outlines and critiques). Outlines are produced with GPT-4o (temperature 1.0, top_p 0.01), o1-preview, and o1. *(`api_client.py`, `config.py`)*
2. **Originality via Semantic Textual Similarity (STS).** Outlines are embedded with Google's `text-embedding-005` model and compared by cosine similarity against (a) non-enhanced outlines and (b) outlines reconstructed from existing literature — 51 relevant sections identified out of 245. A Mann-Kendall trend test is applied to each 11-point STS sequence (non-enhanced through 10 enhancement cycles); lower STS indicates greater originality. *(`embeddings.py`, `visualization.py`, `data_processing.py`)*
3. **Cogency & originality via human survey.** A Qualtrics survey recruited PhD-level Prolific participants in computer science, philosophy, or psychology. Each participant chose the stronger of two outlines on plausibility, support, and originality. Four outline types were compared (constructed-from-existing-section, non-enhanced, originality-enhanced, cogency-and-originality-enhanced), yielding 24 pairs per claim × 2 claims = 48 responses per survey. A Bradley-Terry model estimates preference scores per type and criterion. *(`survey.py`)*
4. **Argument-text generation from outlines.** The best outlines are expanded into ~800-word academic-style argument texts, comparing o1, GPT-4o, and a fine-tuned GPT-4o. Fine-tuning follows a self-instruct-style recipe: 245 cleaned literature sections are revised, broken into outlines, and paired with synthesized system instructions for supervised fine-tuning (plus a "simply fine-tuned" variant using only the generation instruction). *(`finetuning.py`)*

### Results

- **Originality (STS).** Enhancement lowered STS to both baselines, indicating greater originality. The decrease was largest for originality-only enhancement and for the reasoning models (o1-preview showed the largest drops; o1 dropped most sharply on originality-only enhancement), and *full* enhancement outperformed *simple* for the reasoning models. Mann-Kendall tests confirmed statistically significant decreasing trends for the non-simple methods.
- **Cogency & originality (human evaluation).** Human judgments ran opposite to the STS signal: critique-revision enhancement *reduced* perceived cogency **and** originality. Non-enhanced outlines scored highest across plausibility, support, and originality, and even outscored outlines reconstructed from RS&G's and Clark's own sections.
- **Argument generation.** o1 produced the most cogent, original, and human-like academic arguments; default GPT-4o was coherent but simpler and more artificial; both fine-tuning methods degraded quality, yielding shallower, more artificial texts.
- **Takeaway.** Iterative critique-revision raised originality by the STS measure but lowered cogency by human judgment. The most promising route to cogent *and* original arguments was a strong reasoning model (o1) given a good outline; fine-tuning did not help.

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
└── original_notebook/
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
