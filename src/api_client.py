"""
Functions for calling the OpenAI Chat Completions API to generate, critique,
and revise argument outlines.
"""

from __future__ import annotations

from openai import OpenAI


def make_client() -> OpenAI:
    """Return an OpenAI client (reads OPENAI_API_KEY from the environment)."""
    return OpenAI()


def generate_responses(
    client: OpenAI,
    model: str,
    seed: int,
    n: int,
    top_p: float,
    system_prompt: str,
    user_prompt: str,
    process_list: list[str],
    response_list: list[str],
    response_critique_history: list[str],
) -> None:
    """
    Call the Chat Completions API and append results to *process_list* and
    *response_list*.

    Each generated response is recorded both as a full Alpaca-style process
    string and as the raw assistant content.  The first choice is also
    appended to *response_critique_history* for downstream critic/revision
    loops.
    """
    completion = client.chat.completions.create(
        model=model,
        seed=seed,
        n=n,
        top_p=top_p,
        store=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    for i in range(n):
        content = completion.choices[i].message.content
        process = (
            "### Instruction:\n" + system_prompt
            + "\n\n### Input:\n" + user_prompt
            + "\n\n### Response:\n" + content
        )
        process_list.append(process)
        response_list.append(content)
        print(f"\n\nResponse {len(response_list)}:\n\n")
        print(content)
        print(f"({len(content)} characters)")

    first_content = completion.choices[0].message.content
    response_critique_history.append(
        f'\n\n**Reasons 1:** """{first_content}"""\n\n'
    )


def improve_outlines(
    client: OpenAI,
    model: str,
    seed: int,
    num_iterations: int,
    claim: str,
    critic_instruction: str,
    critic_input_template: str,
    revision_instruction: str,
    revision_input_template: str,
    process_list: list[str],
    outlines: list[str],
    response_critique_history: list[str],
    simple_enhance: bool,
) -> None:
    """
    Iteratively critique the latest outline and revise it.

    When *simple_enhance* is False the full critique history is sent to the
    revision step; when True only the current round's history is used.
    """
    for i in range(num_iterations):
        # --- Critique step ---------------------------------------------------
        latest_outline = outlines[-1]
        completion = client.chat.completions.create(
            model=model,
            seed=seed,
            store=True,
            reasoning_effort="high",
            messages=[
                {"role": "developer", "content": critic_instruction},
                {
                    "role": "user",
                    "content": critic_input_template.format(claim, latest_outline),
                },
            ],
        )
        critique_content = completion.choices[0].message.content
        process_list.append(
            "### Instruction:\n" + critic_instruction
            + "\n\n### Input:\n"
            + critic_input_template.format(claim, latest_outline)
            + "\n\n### Response:\n" + critique_content
        )
        response_critique_history.append(
            f'\n\n**Critique {i + 1}** on **Reasons {i + 1}:** '
            f'"""{critique_content}"""\n\n'
        )
        print(f"\n\nCritique {i + 1}:\n\n")
        print(critique_content)

        # --- Revision step ----------------------------------------------------
        if simple_enhance:
            history_text = (
                response_critique_history[2 * i]
                + response_critique_history[2 * i + 1]
            )
        else:
            history_text = "".join(response_critique_history)

        completion = client.chat.completions.create(
            model=model,
            seed=seed,
            store=True,
            reasoning_effort="high",
            messages=[
                {
                    "role": "developer",
                    "content": revision_instruction.format("Task Prompt:"),
                },
                {
                    "role": "user",
                    "content": revision_input_template.format(claim, history_text),
                },
            ],
        )
        revised_content = completion.choices[0].message.content
        process_list.append(
            "### Instruction:\n" + revision_instruction.format("Task Prompt:")
            + "\n\n### Input:\n"
            + revision_input_template.format(claim, history_text)
            + "\n\n### Response:\n" + revised_content
        )
        outlines.append(revised_content)
        response_critique_history.append(
            f'\n\n**Reasons {i + 2}** given **Critique {i + 1}:**\n'
            f'"""{revised_content}"""\n\n'
        )
        print(f"\n\nReasons {i + 2}:\n\n")
        print(revised_content)
