"""
Prompt templates, claim definitions, and background texts used throughout
the argument-generation pipeline.
"""

# ---------------------------------------------------------------------------
# Alpaca-style prompt wrapper
# ---------------------------------------------------------------------------
ALPACA_PROMPT = """

### Instruction:
{}

### Input:
{}

### Response:
{}"""

OUTPUT_TEXT = ""

# ---------------------------------------------------------------------------
# Evaluation criteria (reusable building blocks for prompt construction)
# ---------------------------------------------------------------------------
CRITERIA = """Ensure the following *criteria* are met:"""

PLAUSIBILITY = """
**Plausibility:** The reasons provided should be plausible. For example, do not hallucinate, and do not make up factual information."""

SUPPORT = """
**Support:** The reasons must support the given main claim. For example, stay focused on the main claim without deviating into unrelated claims."""

ORIGINALITY = """
**Originality:** Your reasons should be original and unique. For example, do not reiterate the well-known arguments.

"""

LENGTH_ARGUMENT_OUTLINE = """
**Length:** The reasons should be around 100 words in total, equivalent to the length of an outline for one body section or subsection."""

LENGTH_ARGUMENT_SECTION = """
**Length:** Your argument should be around 800 words, equivalent to the length of one body section or subsection."""

# ---------------------------------------------------------------------------
# Main claims
# ---------------------------------------------------------------------------
CLAIM_RSG = """

**Main Claim:** *Propositional modularity\\* is incompatible with connectionist models\\*\\* of cognitive systems.*

"""

CLAIM_CLARK = """

**Main Claim:** *Functionally discrete\\* beliefs playing a causal role\\* **can** be compatible with connectionist models\\*\\* of cognitive systems.*

"""

CLAIM_SMOLSTICH = """

**Main Claim:** *Although there are no propositionally modular beliefs, eliminativism regarding belief is not justified.*

"""

CLAIM_CHROOM = """

**Main Claim:** *Any digital computer does not understand Chinese soley on the basis of implementing the appropriate program for understanding Chinese.*

"""

# ---------------------------------------------------------------------------
# Background / reference texts
# ---------------------------------------------------------------------------
TEXT_PROP_MODUL = (
    "\"Though common-sense psychology contains a wealth of lore about beliefs, "
    "memories, desires, hopes, fears and the other propositional attitudes, the "
    "crucial folk psychological tenets in forging the link between connectionism "
    "and eliminativism are the claims that propositional attitudes are "
    "*functionally discrete*, *semantically interpretable*, states that play a "
    "*causal role* in the production of other propositional attitudes, and "
    "ultimately in the production of behavior. Following the suggestion in Stich "
    "(1983, pp. 237 ff.), we'll call this cluster of claims *propositional "
    "modularity*.\n\nTo illustrate the way in which folk psychology takes "
    "propositional attitudes to be *functionally discrete*, *causally active* "
    "states, let us sketch a more elaborate example.\n\nIn common-sense "
    "psychology, behavior is often explained by appeal to certain of the agent's "
    "beliefs and desires. Thus, to explain why Alice went to her office, we "
    "might note that she wanted to send some e-mail messages (and, of course, "
    "she believed she could do so from her office). However, in some cases an "
    "agent will have several sets of beliefs and desires each of which might "
    "lead to the same behavior. Thus we may suppose that Alice also wanted to "
    "talk to her research assistant, and that she believed he would be at the "
    "office. In such cases, common sense psychology assumes that Alice's going "
    "to her office might have been caused by either one of the belief/desire "
    "pairs, or by both, and that determining which of these options obtains is "
    "an empirical matter. So it is entirely possible that on this occasion "
    "Alice's desire to send some e-mail played no role in producing her "
    "behavior; it was the desire to talk with her research assistant that "
    "actually caused her to go to the office. However, had she not wanted to "
    "talk with her research assistant, she might have gone to the office anyhow, "
    "because the desire to sent some e-mail, which was causally inert in her "
    "actual decision making, might then have become actively involved. Note that "
    "in this case common sense psychology is prepared to recognize a pair of "
    "quite *distinct* *semantically characterized* states, one of which may be "
    "*causally active* while the other is not.\""
)

TEXT_CONN_MODEL = (
    "\"Connectionist models are large networks of simple parallel computing "
    "elements, each of which carries a numerical *activation value* which it "
    "computes from the values of neighboring elements in the network, using some "
    "simple numerical formula. The network elements, or *units*, influence each "
    "other's values through connections that carry a numerical strength or "
    "*weight* ...\n\nIn a typical ... model, input to the system is provided by "
    "imposing activation values on the *input units* of the network; these "
    "numerical values represent some encoding, or *representation*, of the "
    "input. The activation on the input units propagates along the connections "
    "until some set of activation values emerges on the *output units*; these "
    "activation values encode the output the system has computed from the input. "
    "In between the input and output units there may be other units, often "
    "called *hidden units*, that participate in representing neither the input "
    "nor the output.\n\nThe computation performed by the network in transforming "
    "the input pattern of activity to the output pattern depends on the set of "
    "connection strengths; these weights are usually regarded as encoding the "
    "system's knowledge. In this sense, the connection strengths play the role "
    "of the program in a conventional computer. Much of the allure of the "
    "Connectionist approach is that many connectionist networks program "
    "themselves, that is, they have autonomous procedures for tuning their "
    "weights to eventually perform some specific computation. Such *learning "
    "procedures* often depend on training in which the network is presented with "
    "sample input/output pairs from the function it is supposed to compute. In "
    "learning networks with hidden units, the network itself 'decides' what "
    "computations the hidden units will perform; because these units represent "
    "neither inputs nor outputs, they are never 'told' what their values should "
    "be, even during training.\n\nIn many connectionist models the hidden units "
    "and the output units are assigned a numerical 'bias' which is added into "
    "the calculation determining the unit's activation level. The learning "
    "procedures for such networks typically set both the connection strengths "
    "and the biases. Thus in these networks the system's knowledge is usually "
    "regarded as encoded in *both* the connection strengths and the biases.\""
)

# Pre-computed summaries (originally generated via GPT-4o-mini)
PROP_MODUL_SUMMARY = """

(\\*Information about propositional modularity, functional discreteness, semantic interpretability, and causal role: Propositional modularity refers to the concept that propositional attitudes, such as beliefs and desires, are functionally discrete, semantically interpretable, and play a causal role in behavior. In common-sense psychology, these attitudes are seen as distinct states that can independently influence actions. For instance, Alice's decision to go to her office could be driven by her desire to send emails or to talk to her research assistant. Each desire represents a semantically distinct state, and either could be causally active in prompting her behavior. This illustrates how propositional attitudes are modular, with each having the potential to independently affect outcomes based on their causal roles. (Ramsey, Stich, and Garon, 1995))

"""

CONN_MODEL_SUMMARY = """

(\\*\\*Information about connectionist models: Connectionist models are networks composed of simple, parallel computing units that process information by propagating activation values through connections with varying strengths, or weights. These models transform input data into output through a series of interconnected units, including hidden units that facilitate complex computations without explicit instructions. The network's knowledge is encoded in the connection strengths and biases, which are adjusted through learning procedures. These procedures often involve training with input/output pairs, allowing the network to autonomously tune its weights and biases to perform specific computations. This self-programming capability is a key feature of connectionist models, enabling them to adapt and learn from data. (Ramsey et al., 1995; Smolensky, 1995))

"""

# ---------------------------------------------------------------------------
# Outline generation prompts
# ---------------------------------------------------------------------------
OUTLINE_GENERATION_TASK = """

**{}** State reasons to support the main claim provided.

"""

OUTLINE_FORM = """

**Reason (1):** [Provide the reason supporting the main claim.]

**Reason (2):** [Provide the reason supporting the main claim.]

**Reason (3):** [Provide the reason supporting the main claim, if applicable.]

[Provide additional reasons if applicable.]

"""

OUTLINE_GENERATION_INSTRUCTION = (
    OUTLINE_GENERATION_TASK.format("Task Prompt:")
    + CRITERIA + LENGTH_ARGUMENT_OUTLINE + PLAUSIBILITY + SUPPORT + ORIGINALITY
)

OUTLINE_GENERATION_INPUT = "{}" + PROP_MODUL_SUMMARY + CONN_MODEL_SUMMARY + OUTLINE_FORM

# ---------------------------------------------------------------------------
# Originality critic prompts
# ---------------------------------------------------------------------------
OUTLINE_ORIGINALITY_CRITIC_INSTRUCTION = """

**{}** For the provided reasons, construct a critique that identify any points that lack originality. Your critique should be around 50 words. Do *not* critique, challenge, or comment on the position or stance of the main claim under any circumstances.

"""

ORIGINALITY_CRITIC_FORM = """

**Critique** on **Reasons**

Non-Original Points:
- [Specify any identified non-original point, if applicable.]
- [Add another identified non-original point if applicable.]
[List additional non-original points, if applicable.]

"""

OUTLINE_ORIGINALITY_CRITIC_INPUT = (
    "{}" + PROP_MODUL_SUMMARY + CONN_MODEL_SUMMARY
    + "\n\n**Reasons**: \"\"\"{}\"\"\"\n\n"
    + OUTLINE_ORIGINALITY_CRITIC_INSTRUCTION.format("Task Reminder:")
    + ORIGINALITY_CRITIC_FORM
)

# ---------------------------------------------------------------------------
# Cogency + originality critic prompts
# ---------------------------------------------------------------------------
OUTLINE_COGENTORIGINALITY_CRITIC_INSTRUCTION = """

**{}** For the provided reasons, construct a critique that identify any points that are implausible, fail to support the main claim, or lack originality. Your critique should be around 50 words. Do *not* critique, challenge, or comment on the position or stance of the main claim under any circumstances.

"""

CRITIC_COGENTORIGINALITY_FORM = """

**Critique** on **Reasons**

Implausible Points:
- [Specify any identified implausible point if applicable.]
[List additional implausible points, if applicable.]

Points Not Supporting the Main Claim:
- [Specify any identified point that does not support the main claim if applicable.]
[List additional points not supporting the main claim, if applicable.]

Non-Original Points:
- [Specify any identified non-original point, if applicable.]
[List additional non-original points, if applicable.]

"""

OUTLINE_COGENTORIGINALITY_CRITIC_INPUT = (
    "{}" + PROP_MODUL_SUMMARY + CONN_MODEL_SUMMARY
    + "\n\n**Reasons**: \"\"\"{}\"\"\"\n\n"
    + OUTLINE_COGENTORIGINALITY_CRITIC_INSTRUCTION.format("Task Reminder:")
    + CRITIC_COGENTORIGINALITY_FORM
)

# ---------------------------------------------------------------------------
# Revision prompts
# ---------------------------------------------------------------------------
REVISE_TASK = """

**{}** Revise the last reasons by removing the points identified in all the critique provided. The revised reasons should *support* the main claim.

"""

REVISE_ORIGINALITY_INSTRUCTION = (
    REVISE_TASK.format("Task Prompt:")
    + CRITERIA + LENGTH_ARGUMENT_OUTLINE + ORIGINALITY
)

REVISE_ORIGINALITY_INPUT = (
    "{}" + PROP_MODUL_SUMMARY + CONN_MODEL_SUMMARY
    + "\n\n\"\"\"{}\"\"\"\n\n"
    + REVISE_TASK.format("Task Reminder:") + OUTLINE_FORM
)

REVISE_COGENTORIGINALITY_INSTRUCTION = (
    REVISE_TASK.format("Task Prompt:")
    + CRITERIA + LENGTH_ARGUMENT_OUTLINE + PLAUSIBILITY + SUPPORT + ORIGINALITY
)

REVISE_COGENTORIGINALITY_INPUT = (
    "{}" + PROP_MODUL_SUMMARY + CONN_MODEL_SUMMARY
    + "\n\n\"\"\"{}\"\"\"\n\n"
    + REVISE_TASK.format("Task Reminder:") + OUTLINE_FORM
)

# ---------------------------------------------------------------------------
# Text revision / synthesis prompts
# ---------------------------------------------------------------------------
TEXT_REVISION_INSTRUCTION = "Revise the whole text provided."

TEXT_REVISION_INPUT = "\"\"\"{}\"\"\"\n\n\n\n" + "Reminder: " + TEXT_REVISION_INSTRUCTION

OUTLINE_SYNTHESIS_TASK = (
    "Break down the provided text into the main claim and reasons "
    "that support the claim.\n\n"
)

OUTLINE_SYNTHESIS_INSTRUCTION = (
    OUTLINE_SYNTHESIS_TASK + CRITERIA + LENGTH_ARGUMENT_OUTLINE
)

OUTLINE_SYNTHESIS_INPUT = (
    "\"\"\"{}\"\"\"\n\n"
    + "\n\nReminder: " + OUTLINE_SYNTHESIS_TASK
    + "\n\n**Main Claim:** *[State the main claim concisely.]*"
    + OUTLINE_FORM
)

# ---------------------------------------------------------------------------
# Argument text generation prompts
# ---------------------------------------------------------------------------
ARGUE_GENERATION_INSTRUCTION = (
    "You are an academic assistant specializing in philosophy of cognitive "
    "science. Your role is writing an argument justifying the given main claim, "
    "based on the reasons provided. Your responses should be detailed and "
    "analytical, following the tone of a scholarly article. You adhere to the "
    "style of academic writing, including structured arguments and illustrative "
    "examples. Your argument must be truthful\u2014avoid fabricating facts or "
    "introducing unfounded claims. Maintain a sharp focus on justifying the "
    "given main claim, without deviating into unrelated topics. The response "
    "should be approximately 800 words, corresponding to a single body section "
    "or subsection of an academic paper."
)

# ---------------------------------------------------------------------------
# Instruction synthesis prompt (for fine-tuning data)
# ---------------------------------------------------------------------------
INSTRUCTION_SYNTHESIS_INSTRUCTION = (
    "You are a prompt synthesizer. Given the provided user input and "
    "corresponding assistant output, generate a concise system message that "
    "instructs an assistant to produce similar responses. Your system message "
    "should clearly define the assistant's role, tone, and style so that it "
    "reliably replicates the output when given the same type of user input."
)

INSTRUCTION_SYNTHESIS_INPUT = (
    'User Input:\n"{}"\n\n\nAssistant Output:\n"{}"\n\n\n'
    "[Your synthesized system message here]"
)

# ---------------------------------------------------------------------------
# Default seeds & iteration settings
# ---------------------------------------------------------------------------
SEEDS = [9, 60, 315, 8714, 3171, 1516, 848, 2039]
NUM_ITERATION = 10
