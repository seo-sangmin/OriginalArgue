#!/usr/bin/env python3
"""
Generating Cogent and Original Arguments — main runner.

This script reproduces the full pipeline from the original Jupyter notebook
(v3.6). Each section can be toggled on or off via the ``RUN_*`` flags below.

Set ``GPT_GEN = True`` to call the OpenAI / Vertex AI APIs and generate
new data; set it to ``False`` (the default) to download previously
generated data from Hugging Face and proceed with analysis only.
"""

from __future__ import annotations

import os
import random
import statistics

# ── Google Cloud / Vertex AI setup (needed for embeddings) ────────────────
# Uncomment the following block when running in Google Colab:
# from google.colab import auth
# auth.authenticate_user()
# import gspread
# from google.auth import default
# creds, _ = default()
# import vertexai
# vertexai.init(project="axial-device-305114")

# ── Project modules ───────────────────────────────────────────────────────
from src.config import (
    ALPACA_PROMPT, OUTPUT_TEXT,
    CLAIM_RSG, CLAIM_CLARK, CLAIM_SMOLSTICH, CLAIM_CHROOM,
    PROP_MODUL_SUMMARY, CONN_MODEL_SUMMARY,
    OUTLINE_GENERATION_INSTRUCTION, OUTLINE_GENERATION_INPUT,
    OUTLINE_ORIGINALITY_CRITIC_INSTRUCTION, OUTLINE_ORIGINALITY_CRITIC_INPUT,
    OUTLINE_COGENTORIGINALITY_CRITIC_INSTRUCTION,
    OUTLINE_COGENTORIGINALITY_CRITIC_INPUT,
    REVISE_ORIGINALITY_INSTRUCTION, REVISE_ORIGINALITY_INPUT,
    REVISE_COGENTORIGINALITY_INSTRUCTION, REVISE_COGENTORIGINALITY_INPUT,
    TEXT_REVISION_INSTRUCTION, TEXT_REVISION_INPUT,
    OUTLINE_SYNTHESIS_INSTRUCTION, OUTLINE_SYNTHESIS_INPUT,
    ARGUE_GENERATION_INSTRUCTION,
    INSTRUCTION_SYNTHESIS_INSTRUCTION, INSTRUCTION_SYNTHESIS_INPUT,
    OUTLINE_FORM,
    SEEDS, NUM_ITERATION,
)
from src.api_client import make_client, generate_responses, improve_outlines
from src.data_io import (
    save_and_upload, download_and_process, download_embeddings,
)
from src.data_processing import (
    load_argument_texts, filter_by_word_count,
    print_word_count_stats, find_relevant_indices,
)
from src.embeddings import (
    embed_in_batches, embed_nested_lists,
    average_similarity, similarities_against_reference,
)
from src.visualization import plot_similarity_curves
from src.survey import (
    build_comparison_pairs, generate_html_tables,
    parse_survey_results, merge_match_results,
    compute_bradley_terry,
)
from src.finetuning import (
    build_simple_finetune_dataset, build_instruction_finetune_dataset,
    save_finetune_dataset, validate_finetune_dataset,
    estimate_training_cost,
)

# ═══════════════════════════════════════════════════════════════════════════
# Pipeline flags — set True / False to control which sections run
# ═══════════════════════════════════════════════════════════════════════════
GPT_GEN = False                # True → call APIs; False → use cached data
RUN_OUTLINE_GENERATION = True  # Section 1: generate / download outlines
RUN_EXISTING_TEXTS = True      # Section 2: revise & synthesise existing texts
RUN_EMBEDDINGS = True          # Section 3: embed outlines, compute similarity
RUN_SURVEY = True              # Section 4: build survey & Bradley-Terry
RUN_FINETUNING = True          # Section 5: prepare fine-tuning dataset

# ── Shared state ──────────────────────────────────────────────────────────
REPO_NAME = "Chickward/processes"
EMBED_REPO = "Chickward/embeddings"
ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN", "")

client = make_client() if GPT_GEN else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. IMPROVING ARGUMENT OUTLINES
# ═══════════════════════════════════════════════════════════════════════════

def run_outline_generation():
    """Generate or download normal, original, and cogent-original outlines."""

    seeds = SEEDS
    gpt_model_title = "o1"
    response_critique_history: list[str] = []

    # ── 1a. Normal outlines ───────────────────────────────────────────────
    n = 8
    if GPT_GEN:
        a_process_listlist_normal = []
        process_listlist_normal = []
        for seed in [seeds[7], seeds[6]]:
            for claim in [CLAIM_RSG, CLAIM_CLARK]:
                proc: list[str] = []
                resp: list[str] = []
                generate_responses(
                    client, gpt_model_title, seed, n, 1,
                    OUTLINE_GENERATION_INSTRUCTION,
                    OUTLINE_GENERATION_INPUT.format(claim),
                    proc, resp, response_critique_history,
                )
                if seed == seeds[7]:
                    process_listlist_normal.append(proc)
                else:
                    a_process_listlist_normal.append(proc)

    # Save / download normal outlines
    n = 11
    seed = seeds[7]
    filename = f"{n}_process_listlist_normal_outline_seed{seed}_{gpt_model_title}"
    if GPT_GEN:
        save_and_upload(process_listlist_normal, filename, REPO_NAME, ACCESS_TOKEN)
    process_listlist_normal, response_listlist_normal = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )

    seed = seeds[6]
    filename = f"a_{n}_process_listlist_normal_outline_seed{seed}_{gpt_model_title}"
    if GPT_GEN:
        save_and_upload(a_process_listlist_normal, filename, REPO_NAME, ACCESS_TOKEN)
    a_process_listlist_normal, a_response_listlist_normal = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )

    # ── 1b. Original outlines ────────────────────────────────────────────
    num_iteration = NUM_ITERATION
    n = 1
    if GPT_GEN:
        process_listlist_original = []
        simple_process_listlist_original = []
        for simple_enhance in [False, True]:
            for claim in [CLAIM_RSG, CLAIM_CLARK]:
                for seed in seeds[1:5]:
                    proc = []
                    resp = []
                    response_critique_history = []
                    generate_responses(
                        client, gpt_model_title, seed, n, 1,
                        OUTLINE_GENERATION_INSTRUCTION,
                        OUTLINE_GENERATION_INPUT.format(claim),
                        proc, resp, response_critique_history,
                    )
                    improve_outlines(
                        client, gpt_model_title, seed, num_iteration,
                        claim,
                        OUTLINE_ORIGINALITY_CRITIC_INSTRUCTION.format("Task Prompt:"),
                        OUTLINE_ORIGINALITY_CRITIC_INPUT,
                        REVISE_ORIGINALITY_INSTRUCTION,
                        REVISE_ORIGINALITY_INPUT,
                        proc, resp, response_critique_history,
                        simple_enhance,
                    )
                    if not simple_enhance:
                        process_listlist_original.append(proc)
                    else:
                        simple_process_listlist_original.append(proc)

    filename = f"{num_iteration * 2 + 1}_process_listlist_original_outline_{gpt_model_title}"
    if GPT_GEN:
        save_and_upload(process_listlist_original, filename, REPO_NAME, ACCESS_TOKEN)
    process_listlist_original, response_listlist_original = download_and_process(
        REPO_NAME, filename, remove_critiques=True,
    )

    filename = f"simple_{num_iteration * 2 + 1}_process_listlist_original_outline_{gpt_model_title}"
    if GPT_GEN:
        save_and_upload(simple_process_listlist_original, filename, REPO_NAME, ACCESS_TOKEN)
    simple_process_listlist_original, simple_response_listlist_original = download_and_process(
        REPO_NAME, filename, remove_critiques=True,
    )

    # ── 1c. Cogent-original outlines ─────────────────────────────────────
    n = 1
    if GPT_GEN:
        process_listlist_cogent = []
        simple_process_listlist_cogent = []
        for simple_enhance in [False, True]:
            for claim in [CLAIM_RSG, CLAIM_CLARK]:
                for seed in seeds[1:5]:
                    proc = []
                    resp = []
                    response_critique_history = []
                    generate_responses(
                        client, gpt_model_title, seed, n, 1,
                        OUTLINE_GENERATION_INSTRUCTION,
                        OUTLINE_GENERATION_INPUT.format(claim),
                        proc, resp, response_critique_history,
                    )
                    improve_outlines(
                        client, gpt_model_title, seed, num_iteration,
                        claim,
                        OUTLINE_COGENTORIGINALITY_CRITIC_INSTRUCTION.format("Task Prompt:"),
                        OUTLINE_COGENTORIGINALITY_CRITIC_INPUT,
                        REVISE_COGENTORIGINALITY_INSTRUCTION,
                        REVISE_COGENTORIGINALITY_INPUT,
                        proc, resp, response_critique_history,
                        simple_enhance,
                    )
                    if not simple_enhance:
                        process_listlist_cogent.append(proc)
                    else:
                        simple_process_listlist_cogent.append(proc)

    filename = f"{num_iteration * 2 + 1}_process_listlist_cogentoriginal_outline_{gpt_model_title}"
    if GPT_GEN:
        save_and_upload(process_listlist_cogent, filename, REPO_NAME, ACCESS_TOKEN)
    process_listlist_cogent, response_listlist_cogent = download_and_process(
        REPO_NAME, filename, remove_critiques=True,
    )

    filename = f"simple_{num_iteration * 2 + 1}_process_listlist_cogentoriginal_outline_{gpt_model_title}"
    if GPT_GEN:
        save_and_upload(simple_process_listlist_cogent, filename, REPO_NAME, ACCESS_TOKEN)
    simple_process_listlist_cogent, simple_response_listlist_cogent = download_and_process(
        REPO_NAME, filename, remove_critiques=True,
    )

    # ── 1d. Word-count statistics ────────────────────────────────────────
    word_counts = [
        len(response[i].split())
        for i in range(11)
        for response in response_listlist_original
    ]
    print(f"Median word count: {statistics.median(word_counts)}")
    print(f"Mean word count:   {statistics.mean(word_counts):.2f}")

    # ── 1e. Irrelevant outlines (smolstich & chroom) ─────────────────────
    n = 8
    gpt_model_title = "o1"

    filename = f"{n}_process_listlist_smolstich_outline_{gpt_model_title}"
    if GPT_GEN:
        process_listlist_smolstich = []
        for seed in [seeds[5], seeds[6]]:
            proc = []
            resp = []
            response_critique_history = []
            generate_responses(
                client, gpt_model_title, seed, n, 1,
                OUTLINE_GENERATION_INSTRUCTION,
                CLAIM_SMOLSTICH + PROP_MODUL_SUMMARY + OUTLINE_FORM,
                proc, resp, response_critique_history,
            )
            process_listlist_smolstich.append(proc)
        save_and_upload(process_listlist_smolstich, filename, REPO_NAME, ACCESS_TOKEN)
    process_listlist_smolstich, response_listlist_smolstich = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )

    filename = f"{n}_process_listlist_chroom_outline_{gpt_model_title}"
    if GPT_GEN:
        process_listlist_chroom = []
        for seed in [seeds[5], seeds[6]]:
            proc = []
            resp = []
            response_critique_history = []
            generate_responses(
                client, gpt_model_title, seed, n, 1,
                OUTLINE_GENERATION_INSTRUCTION,
                CLAIM_CHROOM + OUTLINE_FORM,
                proc, resp, response_critique_history,
            )
            process_listlist_chroom.append(proc)
        save_and_upload(process_listlist_chroom, filename, REPO_NAME, ACCESS_TOKEN)
    process_listlist_chroom, response_listlist_chroom = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )

    return {
        "response_listlist_normal": response_listlist_normal,
        "a_response_listlist_normal": a_response_listlist_normal,
        "response_listlist_original": response_listlist_original,
        "simple_response_listlist_original": simple_response_listlist_original,
        "response_listlist_cogent": response_listlist_cogent,
        "simple_response_listlist_cogent": simple_response_listlist_cogent,
        "response_listlist_smolstich": response_listlist_smolstich,
        "response_listlist_chroom": response_listlist_chroom,
        "process_listlist_normal": process_listlist_normal,
        "a_process_listlist_normal": a_process_listlist_normal,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXISTING ARGUMENT TEXTS — revise, synthesise outlines, find relevant
# ═══════════════════════════════════════════════════════════════════════════

def run_existing_texts():
    """Load, filter, revise existing texts and synthesise outlines."""

    seeds = SEEDS

    # Load & filter
    texts = load_argument_texts()
    print_word_count_stats(texts)
    texts = filter_by_word_count(texts)

    num_samples = len(texts)
    indices = list(range(num_samples))

    # Revise existing texts
    seed = seeds[0]
    gpt_model_title = "gpt-4o"
    filename = f"{num_samples}_process_revised_existing_texts_seed{seed}_{gpt_model_title}"

    if GPT_GEN:
        response_critique_history: list[str] = []
        proc_revised: list[str] = []
        resp_revised: list[str] = []
        n = 1
        for i in indices:
            generate_responses(
                client, gpt_model_title, seed, n, 0.01,
                TEXT_REVISION_INSTRUCTION,
                TEXT_REVISION_INPUT.format(texts[i]),
                proc_revised, resp_revised, response_critique_history,
            )
        save_and_upload(proc_revised, filename, REPO_NAME, ACCESS_TOKEN)

    _, response_revised = download_and_process(REPO_NAME, filename, remove_critiques=False)
    response_revised = [
        t.replace("**Revised Text:**", "", 1).strip() for t in response_revised
    ]

    # Synthesise outlines
    seed = seeds[0]
    gpt_model_title = "gpt-4o"
    filename = f"{len(texts)}_process_synthesis_outline_seed{seed}_{gpt_model_title}"

    if GPT_GEN:
        response_critique_history = []
        proc_synth: list[str] = []
        resp_synth: list[str] = []
        n = 1
        for text in response_revised:
            generate_responses(
                client, gpt_model_title, seed, n, 0.01,
                OUTLINE_SYNTHESIS_INSTRUCTION,
                OUTLINE_SYNTHESIS_INPUT.format(text),
                proc_synth, resp_synth, response_critique_history,
            )
        save_and_upload(proc_synth, filename, REPO_NAME, ACCESS_TOKEN)

    _, gpt_response_synthesis_outline = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )

    # Find relevant texts
    relevant_indices = find_relevant_indices(response_revised)
    most_relevant = [196, 83, 91, 94]
    check = all(i in relevant_indices for i in most_relevant)
    print(f"Most-relevant indices included: {check}")

    # Synthesise outlines for relevant texts
    gpt_model_title = "o1"
    seed = seeds[0]
    filename = f"{len(relevant_indices)}_process_relevant_outline_seed{seed}_{gpt_model_title}"

    if GPT_GEN:
        response_critique_history = []
        proc_rel: list[str] = []
        resp_rel: list[str] = []
        n = 1
        for i in relevant_indices:
            generate_responses(
                client, gpt_model_title, seed, n, 1,
                OUTLINE_SYNTHESIS_INSTRUCTION,
                OUTLINE_SYNTHESIS_INPUT.format(texts[i]),
                proc_rel, resp_rel, response_critique_history,
            )
        save_and_upload(proc_rel, filename, REPO_NAME, ACCESS_TOKEN)

    _, gpt_relevant_existing_outlines = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )
    print(f"Relevant existing outlines: {len(gpt_relevant_existing_outlines)}")

    return {
        "texts": texts,
        "response_revised": response_revised,
        "gpt_response_synthesis_outline": gpt_response_synthesis_outline,
        "gpt_relevant_existing_outlines": gpt_relevant_existing_outlines,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. EMBEDDINGS & SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def run_embeddings(outlines: dict, existing: dict):
    """Embed outlines, compute similarities, and plot trends."""

    seeds = SEEDS
    gpt_model_title = "o1"

    # Embed normal outlines (always computed fresh via Vertex AI)
    normal_emb = embed_nested_lists(outlines["response_listlist_normal"], 12, 10)
    a_normal_emb = embed_nested_lists(outlines["a_response_listlist_normal"], 12, 10)
    smolstich_emb = embed_nested_lists(outlines["response_listlist_smolstich"], 12, 10)
    chroom_emb = embed_nested_lists(outlines["response_listlist_chroom"], 12, 10)

    # Download pre-computed enhanced-outline embeddings
    def _dl(name):
        return download_embeddings(EMBED_REPO, name)

    n_orig = len(outlines["response_listlist_original"])
    original_emb = _dl(f"{n_orig}_response_listlist_original_outline_embeddings_{gpt_model_title}")
    simple_original_emb = _dl(f"simple_{n_orig}_response_listlist_original_outline_embeddings_{gpt_model_title}")

    n_cog = len(outlines["response_listlist_cogent"])
    cogent_emb = _dl(f"{n_cog}_response_listlist_cogentoriginal_outline_embeddings_{gpt_model_title}")
    simple_cogent_emb = _dl(f"simple_{n_cog}_response_listlist_cogentoriginal_outline_embeddings_{gpt_model_title}")

    # Existing-outline embeddings
    gpt_model_4o = "gpt-4o"
    seed = seeds[0]
    n_synth = len(existing["gpt_response_synthesis_outline"])
    existing_emb = _dl(f"{n_synth}_existing_outlines_embeddings_seed{seed}_{gpt_model_4o}")

    n_rel = len(existing["gpt_relevant_existing_outlines"])
    relevant_emb = _dl(f"{n_rel}_existing_outlines_embeddings_{gpt_model_title}")

    # ── Compute similarities ─────────────────────────────────────────────
    print("=== Similarities vs. normal outlines ===")
    sim_normal_normal = similarities_against_reference(a_normal_emb, normal_emb)
    sim_orig_normal = similarities_against_reference(original_emb, normal_emb)
    sim_cogent_normal = similarities_against_reference(cogent_emb, normal_emb)
    simple_sim_orig_normal = similarities_against_reference(simple_original_emb, normal_emb)
    simple_sim_cogent_normal = similarities_against_reference(simple_cogent_emb, normal_emb)
    sim_smolstich_normal = similarities_against_reference(smolstich_emb, normal_emb)

    print("\n=== Similarities vs. relevant existing outlines ===")
    sim_normal_relevant = similarities_against_reference(a_normal_emb, relevant_emb)
    sim_orig_relevant = similarities_against_reference(original_emb, relevant_emb)
    sim_cogent_relevant = similarities_against_reference(cogent_emb, relevant_emb)
    simple_sim_orig_relevant = similarities_against_reference(simple_original_emb, relevant_emb)
    simple_sim_cogent_relevant = similarities_against_reference(simple_cogent_emb, relevant_emb)
    sim_chroom_relevant = similarities_against_reference(chroom_emb, relevant_emb)

    # Flatten baselines to constant lines
    avg_nn = sum(sim_normal_normal) / len(sim_normal_normal)
    sim_normal_normal = [avg_nn] * 11

    avg_nr = sum(sim_normal_relevant) / len(sim_normal_relevant)
    sim_normal_relevant = [avg_nr] * 11

    avg_sn = sum(sim_smolstich_normal) / len(sim_smolstich_normal)
    sim_smolstich_normal = [avg_sn] * 11

    avg_cr = sum(sim_chroom_relevant) / len(sim_chroom_relevant)
    sim_chroom_relevant = [avg_cr] * 11

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_similarity_curves(
        {
            "Non-Enhanced": sim_normal_normal,
            "Originality Enhanced Simply": simple_sim_orig_normal,
            "Originality Enhanced": sim_orig_normal,
            "Cogency and Originality Enhanced Simply": simple_sim_cogent_normal,
            "Cogency and Originality Enhanced": sim_cogent_normal,
            "Irrelevant": sim_smolstich_normal,
        },
        y_limits=(0.88, 0.985),
        caption="Mann-Kendall Test on STS to Non-Enhanced Outlines",
    )

    plot_similarity_curves(
        {
            "Non-Enhanced": sim_normal_relevant,
            "Originality Enhanced Simply": simple_sim_orig_relevant,
            "Originality Enhanced": sim_orig_relevant,
            "Cogency and Originality Enhanced Simply": simple_sim_cogent_relevant,
            "Cogency and Originality Enhanced": sim_cogent_relevant,
            "Irrelevant": sim_chroom_relevant,
        },
        y_limits=(0.75, 0.81),
        caption="Mann-Kendall Test on Similarity to Outlines of the Relevant Existing Arguments",
    )

    return {
        "a_normal_emb": a_normal_emb,
        "original_emb": original_emb,
        "cogent_emb": cogent_emb,
        "relevant_emb": relevant_emb,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. SURVEY — comparison pairs & Bradley-Terry
# ═══════════════════════════════════════════════════════════════════════════

def run_survey(outlines: dict, emb_data: dict):
    """Build survey pairs, generate HTML, and compute Bradley-Terry scores."""

    # The survey section relies on Google Sheets data.  Uncomment and
    # provide ``creds`` when running in a Colab environment with
    # authenticated Google account.

    # import gspread
    # gc = gspread.authorize(creds)
    # ws1 = gc.open('Evaluating Argument Outlines_January 27, 2025_22.07').worksheet('Sheet2')
    # ws2 = gc.open('Evaluating Argument Outlines (o1-preview) with Comprehension Check_February 4, 2025_08.37').worksheet('Sheet2')
    # rows1 = ws1.get_all_values()
    # rows2 = ws2.get_all_values()
    # match_results1 = parse_survey_results(rows1)
    # match_results2 = parse_survey_results(rows2)
    # combined = merge_match_results(match_results1, match_results2)
    # for matches in combined.values():
    #     print(matches)
    #     compute_bradley_terry(matches)

    print("Survey section: enable Google Sheets access for full execution.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. FINE-TUNING DATA
# ═══════════════════════════════════════════════════════════════════════════

def run_finetuning(existing: dict):
    """Prepare, validate, and estimate cost for fine-tuning datasets."""

    outlines = existing["gpt_response_synthesis_outline"]
    revised = existing["response_revised"]

    # Simple fine-tuning dataset
    dataset = build_simple_finetune_dataset(
        ARGUE_GENERATION_INSTRUCTION, outlines, revised,
    )
    print(f"Simple fine-tuning examples: {len(dataset)}")

    # Instruction fine-tuning dataset (requires synthesised instructions)
    seeds = SEEDS
    seed = seeds[5]
    gpt_model_title = "gpt-4o"
    filename = (
        f"topp0.01_{len(outlines)}_process_synthesis_instruction"
        f"_seed{seed}_{gpt_model_title}"
    )

    if GPT_GEN:
        response_critique_history: list[str] = []
        proc_instr: list[str] = []
        resp_instr: list[str] = []
        n = 1
        for i in range(len(outlines)):
            generate_responses(
                client, gpt_model_title, seed, n, 0.01,
                INSTRUCTION_SYNTHESIS_INSTRUCTION,
                INSTRUCTION_SYNTHESIS_INPUT.format(outlines[i], revised[i]),
                proc_instr, resp_instr, response_critique_history,
            )
        save_and_upload(proc_instr, filename, REPO_NAME, ACCESS_TOKEN)

    _, response_synthesis_instruction = download_and_process(
        REPO_NAME, filename, remove_critiques=False,
    )

    dataset = build_instruction_finetune_dataset(
        response_synthesis_instruction, outlines, revised,
    )
    print(f"Instruction fine-tuning examples: {len(dataset)}")

    save_finetune_dataset(dataset)
    validate_finetune_dataset(dataset)
    estimate_training_cost(dataset)

    # Upload to OpenAI for fine-tuning
    if GPT_GEN:
        client.files.create(
            file=open("gpt_finetune_dataset.jsonl", "rb"),
            purpose="fine-tune",
        )

    # Generate styled argument texts with fine-tuned model
    if GPT_GEN:
        ft_model = "ft:gpt-4o-2024-08-06:personal:instruction-outline-revisedconneli:B1gnWTaX"
        seed = seeds[6]
        response_critique_history = []
        proc_styled: list[str] = []
        resp_styled: list[str] = []
        # Use a cogent-original best outline as input
        generate_responses(
            client, ft_model, seed, 1, 0.01,
            ARGUE_GENERATION_INSTRUCTION,
            "",  # would be the best cogent-original outline
            proc_styled, resp_styled, response_critique_history,
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    outlines = {}
    existing = {}
    emb_data = {}

    if RUN_OUTLINE_GENERATION:
        print("\n" + "=" * 60)
        print("SECTION 1: Improving Argument Outlines")
        print("=" * 60)
        outlines = run_outline_generation()

    if RUN_EXISTING_TEXTS:
        print("\n" + "=" * 60)
        print("SECTION 2: Existing Argument Texts")
        print("=" * 60)
        existing = run_existing_texts()

    if RUN_EMBEDDINGS and outlines and existing:
        print("\n" + "=" * 60)
        print("SECTION 3: Embeddings & Similarity")
        print("=" * 60)
        emb_data = run_embeddings(outlines, existing)

    if RUN_SURVEY and outlines:
        print("\n" + "=" * 60)
        print("SECTION 4: Survey & Bradley-Terry")
        print("=" * 60)
        run_survey(outlines, emb_data)

    if RUN_FINETUNING and existing:
        print("\n" + "=" * 60)
        print("SECTION 5: Fine-Tuning")
        print("=" * 60)
        run_finetuning(existing)

    print("\nDone.")


if __name__ == "__main__":
    main()
