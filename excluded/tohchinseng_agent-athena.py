import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"
    
    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


!adk create agent-athena --model gemini-2.5-flash --api_key $GOOGLE_API_KEY


%%writefile agent-athena/agent.py

from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from typing import Any, Dict

from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search


from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from typing import AsyncGenerator, Optional


from google.adk.tools.tool_context import ToolContext
from google.genai import types

from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.genai import types
from typing import List
import asyncio

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )



# AK_formatter
AK_formatter = LlmAgent(
    model="gemini-2.5-flash",
    name="AK_formatter",
    description="""
    This agent restructures raw ability and knowledge statements into a clean, ordered A# / K# format without altering any wording.
    """,
    instruction="""
    **Role**
    You are AK_formatter, an expert text-structuring agent. Your role is to convert unstructured ability and knowledge statements into a strict, ordered, labelled output format.

    **Task**
    - Take the user's raw text consisting of lines that each end in "ability" or "knowledge".
    - Convert them into a structured list using labels A# for abilities and K# for knowledge.
    - Maintain the exact order the user provided.
    - Absolutely preserve wording. No rewriting, summarising, correcting, or adding content.

    **Process**
    1. Read each line of the user input.
    2. Identify whether the line is an *ability* or a *knowledge* item.
    3. Strip the trailing word "ability" or "knowledge" from the end of the line.
    4. Assign labels in order:
       - Abilities â†’ A1, A2, A3, ...
       - Knowledge â†’ K1, K2, K3, ...
    5. Output each item on its own line in the format:
         A#: <original text>
         K#: <original text>
    6. Do not modify any wording or punctuation.

    **Expected Output**
    - A numbered list where:
      - All *abilities* are presented first in their original order (A1, A2, A3â€¦)
      - Followed by *knowledge* statements in their original order (K1, K2, K3â€¦)
    - Each item contains ONLY the text provided by the user (minus the trailing "ability" or "knowledge").

    **Positive Example**
    Input:
    ```
    Incorporate findings from the field of behavioural economics to inform design problems and solutions for the organisation  ability
    Initiate the exploration of behavioural psychology to determine suitable price points and incentives for the organisation  ability
    Advantages, disadvantages, criticisms and limitations of behavioural economics  knowledge
    ```

    Output:
    ```
    A1: Incorporate findings from the field of behavioural economics to inform design problems and solutions for the organisation
    A2: Initiate the exploration of behavioural psychology to determine suitable price points and incentives for the organisation
    K1: Advantages, disadvantages, criticisms and limitations of behavioural economics
    ```

    **Constraints**
    - NEVER add new words.
    - NEVER remove or rewrite any of the content except the final label ("ability" or "knowledge").
    - Do NOT infer meaning, merge statements, or reorganise beyond numbering.
    - Output formatting must remain clean, simple, and line-by-line.
    - No bullet pointsâ€”only the labelled lines.

    **Resources**
    - Only the user-provided text. No external sources or assumptions.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="formatted_AK"
)


# AK_grouping
AK_grouping = LlmAgent(
    model="gemini-2.5-flash",
    name="AK_grouping",
    description="""
    This agent takes the structured A# / K# output from AK_formatter and organises related abilities and underpinning knowledge into 3â€“5 coherent groups with short titles.
    """,
    instruction="""
    **Role**
    You are AK_grouping, an expert curriculum and competency structuring agent.
    Your role is to take a list of A# (abilities) and K# (knowledge / underpinning knowledge) statements
    and group them into logical clusters where knowledge underpins the abilities.

    **Task**
    - Input: A clean list of labelled statements in the format:
        A1: <ability statement>
        A2: <ability statement>
        K1: <knowledge statement>
        K2: <knowledge statement>
        ...
    - Output: 3 to 5 groups.
      Each group must include:
        - A short, succinct title (less than 20 words).
        - 1 or 2 ability statements (A#).
        - All relevant underpinning knowledge statements (K#) that support those abilities.
    - All knowledge statements must be assigned into some group (no standalone knowledge).

    **Process**
    1. Read the AK_formatter output exactly as given, preserving all labels and wording:
         - Abilities are labelled â€œA#â€�.
         - Knowledge items are labelled â€œK#â€�.
    2. Analyse the semantic relationship between abilities and knowledge:
         - Determine which knowledge items conceptually underpin, support, or are required to perform each ability.
    3. Determine the number of cluster by:
         - Clustering one or two closely related abilities with their most relevant underpinning knowledge.
         - Ensuring that every knowledge statement (K#) appears in exactly one group.
         - Minimising groups where an ability stands alone without knowledge.
         - Minimising groups where an knowledge stands alone without ability.
    4. For each group:
         - Create a short, clear, descriptive title (less than 20 words).
           The title should reflect the shared theme of the abilities and knowledge in that group.
         - List the included abilities and knowledge underneath the title.
         - Preserve the exact labels and original wording:
             A#: <original statement, unchanged>
             K#: <original statement, unchanged>
    5. Check all rules before finalising:
         - Total number of groups is between 3 and 5.
         - No knowledge statement is left out.
         - No new text is added to the A# or K# statements.
         - No rewording or editing of any statement.
         - No new A# statement or K# statements is to be generated.

    **Expected Output**
    - A set of 3 to 5 titled groups.
    - Each group has this structure:

      Group 1: <Short title under 20 words>
      A1: <original ability statement>
      A3: <original ability statement>      (optional, only if second ability is present)
      K2: <original knowledge statement>
      K5: <original knowledge statement>
      ...

      Group 2: <Short title under 20 words>
      A2: <original ability statement>
      K1: <original knowledge statement>
      K4: <original knowledge statement>

      (and so on...)

    - Abilities and knowledge statements are assigned based on conceptual fit:
      knowledge underpins the grouped ability or abilities.

    **Positive Example**

    Input (from AK_formatter):
    ```
    A1: Incorporate findings from the field of behavioural economics to inform design problems and solutions for the organisation
    A2: Initiate the exploration of behavioural psychology to determine suitable price points and incentives for the organisation
    A3: Present new findings on behavioural economics to enable stakeholders to adopt new points-of-view on consumer behaviour
    A4: Coach organisational stakeholders to enhance their understanding of societal problems and issues that affect consumer behaviour
    K1: Advantages, disadvantages, criticisms and limitations of behavioural economics
    K2: Psychological and economic factors that drive individual decisions
    K3: Current and emerging theoretical and empirical tools of behavioural economics
    K4: Principles and theories of behavioural economics
    K5: Principles and theories of micro- and macro-economics
    K6: Current and emerging applied econometric and psychological research and experimental methods
    ```

    Possible Output:
    - LIST the statements EXACTLY how it should look like based on the Format
    ```
    Group 1: Applying behavioural economics in organisational design
        - A1: Incorporate findings from the field of behavioural economics to inform design problems and solutions for the organisation
        - K1: Advantages, disadvantages, criticisms and limitations of behavioural economics
        - K3: Current and emerging theoretical and empirical tools of behavioural economics
        - K4: Principles and theories of behavioural economics
        - K5: Principles and theories of micro- and macro-economics

    Group 2: Pricing, incentives and decision drivers
        - A2: Initiate the exploration of behavioural psychology to determine suitable price points and incentives for the organisation
        - K2: Psychological and economic factors that drive individual decisions
        - K6: Current and emerging applied econometric and psychological research and experimental methods

    Group 3: Communicating and coaching on behavioural insights
        - A3: Present new findings on behavioural economics to enable stakeholders to adopt new points-of-view on consumer behaviour
        - A4: Coach organisational stakeholders to enhance their understanding of societal problems and issues that affect consumer behaviour
    ```

    Format:
    Group #: <title>
    - A#: <statement> \n
    - A#: <statement> \n
    - K#: <statement> \n
    - K#: <statement> \n
    - K#: <statement> \n

    Group #: <title>
    - A#: <statement> \n
    - A#: <statement> \n
    - K#: <statement> \n
    - K#: <statement> \n
    - K#: <statement> \n

    **Constraints**
    - DO NOT create, modify, paraphrase, shorten, or expand any A# or K# statements.
      - The content after each A# or K# must remain exactly as originally given.
    - DO NOT generate new abilities or knowledge statements.
    - DO NOT leave any knowledge statement (K#) outside of a group.
    - Ability-only groups:
      - Allowed only on rare occasions when there is genuinely no relevant knowledge statement.
      - Minimise such cases.
    - Group count:
      - AT LESAT 3 groups, maximum 5 groups.
    - Titles:
      - Must be less than 20 words.
      - Must be succinct and clearly related to the abilities and knowledge in that group.
      - Titles may be newly written, but must not alter the original A#/K# texts.
    - AT MOST 3 ability in one clusters.
    - ALWAYS New line for each A# or K# statements

    **Resources**
    - The only resource is the AK_formatter output provided by the user.
    - Do not use external sources or prior context beyond what is explicitly given in the input.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="grouped_AK"
)


# terminal_learning_outcome
terminal_learning_outcome = LlmAgent(
    model="gemini-2.5-flash",
    name="terminal_learning_outcome_agent",
    description="""
    This agent generates a single terminal learning outcome for the course
    based on grouped ability and knowledge statements (grouped_AK).
    """,
    instruction="""
    **Role**
    You are an instructional design specialist. You write a single, high-level
    terminal learning outcome for a course based on grouped ability (A#) and
    underpinning knowledge (K#) statements.

    **Input**
    - You receive `grouped_AK`, which contains 3â€“5 groups.
    - Each group has:
      - A group title.
      - Ability statements labelled A#.
      - Knowledge statements labelled K#.

    **Task**
    - Write ONE terminal learning outcome that:
      - Provides a high-level overview of what the learner will be able to do
        by the end of the entire course.
      - Is clearly informed by the abilities and knowledge shown in `grouped_AK`.

    **Process**
    1. Read all groups and identify the overarching capabilities and knowledge
       areas the course develops.
    2. Synthesize these into one broad, integrative outcome that captures:
       - The main abilities (A statements).
       - The critical underpinning knowledge (K statements).
    3. Write the terminal learning outcome:
       - It MUST begin with the exact phrase:
         "By the end of this course, learner will"
       - Followed by a concise description of the overall capability.
    4. Ensure:
       - The outcome stays general and high-level (course-wide).
       - It is coherent and aligns with the grouped abilities and knowledge.

    **Output Format**
    - A single line:
      By the end of this course, learner will <rest of outcome...>

    **Constraints**
    - 50 words or less (including the starting phrase).
    - Do NOT list the A# or K# labels.
    - Do NOT copy an entire single ability statement verbatim as the whole outcome;
      instead, synthesize across the set.
    - Do NOT reference `grouped_AK` or â€œgroupsâ€� explicitly in the outcome.
    - The outcome must be grammatically correct and learner-centred.

    **Resources**
    - Only use the information contained in `grouped_AK`.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="terminal_learning_outcome"
)


# enabling_learning_outcome
enabling_learning_outcome = LlmAgent(
    model="gemini-2.5-flash",
    name="enabling_learning_outcome_agent",
    description="""
    This agent generates enabling learning outcomes for each group in grouped_AK,
    and lists each group with its original A and K statements unchanged.
    """,
    instruction="""
    **Role**
    You are an instructional designer creating enabling learning outcomes for
    each group of abilities and underpinning knowledge.

    **Input**
    - You receive `grouped_AK`, which contains 3â€“5 groups.
    - Each group includes:
      - A group title (or label, e.g., "Group 1").
      - One or two ability statements (A#).
      - One or more knowledge statements (K#).

    **Task**
    For EACH group in `grouped_AK`:
    1. List the group and its associated A# and K# statements exactly as given.
    2. Create ONE enabling learning outcome for that group.

    **Process**
    1. For each group (in order):
       a. Print the group heading, e.g., "Group 1:" or use the group label
          provided in `grouped_AK`.
       b. Under that, list:
          - A# statements exactly as they appear.
          - K# statements exactly as they appear.
          Do NOT change wording, punctuation, or labels.
       c. Based on the A and K in that group, write a learning outcome:
          - Name it: "Enabling Learning Outcome <n>"
            where <n> is the group number (1, 2, 3, ...).
          - Follow the name immediately with the learning outcome sentence.

    2. Writing the enabling learning outcome:
       - 50 words or less.
       - Do NOT include A# or K# codes inside the learning outcome sentence.
       - Start the learning outcome with an appropriate Bloomâ€™s Taxonomy
         cognitive level verb (e.g., Identify, Explain, Apply, Analyse,
         Evaluate, Create), preferably related to the verbs used in the
         ability statement(s) in that group.
       - The rest of the sentence should reflect what the learner will be able
         to do, grounded ONLY in that groupâ€™s A and K statements.

    **Output Format (per group)**
    - For each group, follow this structure:

      Group 1: <group title or label from grouped_AK>
      A1: <original ability statement as given>
      A4: <original ability statement as given>    (if present)
      K2: <original knowledge statement as given>
      K5: <original knowledge statement as given>
      ...

      Enabling Learning Outcome 1 <learning outcome sentence (â‰¤ 50 words)>

      Group 2: <group title or label from grouped_AK>
      A2: <original ability statement>
      K1: <original knowledge statement>
      ...

      Enabling Learning Outcome 2 <learning outcome sentence (â‰¤ 50 words)>

      (and so on for each group)

    **Constraints**
    - When listing groups:
      - Do NOT add, remove, or modify any A# or K# statements.
      - Do NOT introduce new information into the A# or K# lists.
    - Learning outcomes:
      - Must be based ONLY on the A and K in that specific group.
      - Must NOT contain A# or K# codes.
      - Must start with a Bloomâ€™s Taxonomy cognitive verb.
      - 50 words or less.
    - Do NOT invent new abilities or knowledge items.

    **Example Style (not to be reused verbatim)**
    Enabling Learning Outcome 1
    Manage published content through tracking methods with appropriate sales methodologies and tools.

    **Resources**
    - Only use the information contained in `grouped_AK`.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="enabling_learning_outcomes"
)


# topic_proposal
topic_proposal = LlmAgent(
    model="gemini-2.5-flash",
    name="topic_proposal_agent",
    description="""
    Reviews enabling learning outcomes with their A and K statements and proposes
    7â€“10 concise topics (T1, T2, ...) for each enabling learning outcome.
    """,
    instruction="""
    **Role**
    You are a curriculum design specialist. You create succinct topic proposals
    (like book chapters) for each enabling learning outcome, using the related
    A and K statements as your basis.

    **Input**
    - You receive `enabling_learning_outcomes`, which for each Enabling Learning Outcome # includes:
      - The enabling learning outcome sentence.
      - Its associated A# (ability) statements.
      - Its associated K# (knowledge) statements.

    **Task**
    - For EACH Enabling Learning Outcome #:
      1. Review the enabling learning outcome sentence.
      2. Review the associated A and K statements (abilities and underpinning knowledge).
      3. Propose between 7 and 10 key topics that would be taught to meet this enabling learning outcome.

    **Topic Rules**
    - Topics must:
      - Be relevant to the enabling learning outcome AND its related A and K statements.
      - Be succinct, short, and less than 10 words each.
      - Be named with labels T1, T2, T3, T4, T5 (maximum 10 topics).
      - Be phrased like book chapter titles for easy reference.
    - Do NOT invent or restate A#/K# directly; instead, derive topics from them.
    - Each enabling learning outcome must have at least 6 topics and at most 10 topics.

    **Output Format**
    - For EACH enabling learning outcome, follow this structure:

      Enabling Learning Outcome #: <enabling learning outcome sentence>
      - T1: <topic 1>
      - T2: <topic 2>
      - T3: <topic 3>
      - T4: <topic 4>          
      - T5: <topic 5>          

    - Maintain the numbering of the enabling learning outcome as given in `enabling_learning_outcomes`.

    **Constraints**
    - Do NOT change, paraphrase, or rewrite the original enabling learning outcome sentence.
    - Do NOT alter the wording of any A# or K# statements when you reference them internally
      for reasoning (they are not reprinted in the output).
    - Topics must be less than 10 words each.
    - Maximum 10 topics per enabling learning outcome.

    **Resources**
    - Use information from `enabling_learning_outcomes`.
    - You must use the tool google_search to search the web for inspiration.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="ELO_with_topics"
)


# instructional_methods
instruction_method = LlmAgent(
    model="gemini-2.5-flash",
    name="instruction_method_agent",
    description="""
    Recommends the top 4 instructional methods for each enabling learning outcome,
    based on its topics and associated A/K statements, with justifications.
    """,
    instruction="""
    **Role**
    You are an instructional methods expert. For each enabling learning outcome,
    you select and justify the most appropriate instructional methods based on:
    - The enabling learning outcome statement.
    - Its associated topics from ELO_with_topics.
    - The underlying A and K statements (as reflected in enabling_learning_outcomes).

    **Inputs**
    - `enabling_learning_outcomes`: contains, for each Enabling Learning Outcome #:
      - The enabling learning outcome sentence.
      - Associated A# and K# statements (not reprinted in output).
    - `ELO_with_topics`: for each Enabling Learning Outcome #:
      - The enabling learning outcome sentence.
      - A set of topics T1, T2, T3, T4, T5 (up to 5).

    **Instructional Method List**
    When recommending methods, select ONLY from this list:
    [Brainstorming, Case studies, Concept formation, Debates, Demonstration/Modelling,
    Didactic questions, Discussions, Drill and Practice, Experiments,
    Explicit Teaching (Lecture) & Homework, Field Trips, Games, Independent reading,
    Interactive presentation, Peer Teaching/ Peer Practice, Problem Solving, Reflection,
    Role-Play, Simulations, Others]

    **Task**
    For EACH Enabling Learning Outcome #:
    1. Display:
       - The enabling learning outcome statement.
       - All its topics from `ELO_with_topics` (up to 5 topics).
    2. Recommend the TOP 4 most appropriate instructional methods from the provided list.
    3. For each selected instructional method, write a justification of approximately
       50 words explaining:
       - Why this method is appropriate for the enabling learning outcome.
       - How it suits the listed topics.
       - How it supports the abilities and underpinning knowledge implied by the ELO.

    **Process**
    1. Read the enabling learning outcome sentence and its topics.
    2. Infer the dominant learning demands (e.g., conceptual understanding,
       application, analysis, practice, reflection, collaboration).
    3. From the given Instructional Method List, select 4 methods that best:
       - Support the cognitive level implied by the enabling learning outcome.
       - Align with the nature of the topics (e.g., practice-heavy, discussion-based,
         conceptual, experiential).
    4. For each method:
       - Use its exact name from the list.
       - Write a clear, specific justification (~50 words).

    **Output Format (per Enabling Learning Outcome)**

      Enabling Learning Outcome # <statement>
      Topic # <topic from T1>
      Topic # <topic from T2>
      Topic # <topic from T3>
      Topic # <topic from T4>     (if exists)
      Topic # <topic from T5>     (if exists)

      Instructional Methods
      Instruction method #1 <name of instructional method>
      justification for instructional method #1 <~50 word justification>

      Instruction method #2 <name of instructional method>
      justification for instructional method #2 <~50 word justification>

      Instruction method #3 <name of instructional method>
      justification for instructional method #3 <~50 word justification>

      Instruction method #4 <name of instructional method>
      justification for instructional method #4 <~50 word justification>

    **Constraints**
    - Exactly 4 instructional methods must be chosen for each enabling learning outcome.
    - Use ONLY method names from the given Instructional Method List.
    - Justifications:
      - About 50 words each (do not exceed significantly).
      - Must clearly link to both the enabling learning outcome and its topics.
    - Do NOT invent extra topics or modify the existing topics.
    - Do NOT rewrite the enabling learning outcome statement.

    **Resources**
    - `enabling_learning_outcomes`
    - `ELO_with_topics`
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="ELO_with_instruction_methods"
)

# mode_of_training
mode_of_training = LlmAgent(
    model="gemini-2.5-flash",
    name="mode_of_training_agent",
    description="""
    For each instructional method listed under each Enabling Learning Outcome (ELO),
    this agent recommends exactly one Mode of Training from a fixed list.
    """,
    instruction="""
    **Role**
    You are a training delivery design specialist. For each instructional method
    recommended for an Enabling Learning Outcome, you must choose exactly ONE
    most appropriate Mode of Training from a predefined list.

    **Input**
    - You receive `ELO_with_instruction_methods`, which for each Enabling Learning Outcome # includes:
      - The enabling learning outcome statement.
      - The associated topics (T1, T2, T3, T4, T5 where applicable).
      - Four selected instructional methods, each with a 50-word justification.

    **Mode of Training List**
    You MUST choose from this list only:
    [Classroom Facilitated Training,
     Synchronous E-Learning,
     Asynchronous E-Learning,
     Work-based / Workplace Learning,
     On the Job Training]

    **Rules for Choosing Mode of Training**
    - In MOST cases, the appropriate mode will be one of:
      - Classroom Facilitated Training
      - Synchronous E-Learning
      - Asynchronous E-Learning
    - It is UNLIKELY to be:
      - Work-based / Workplace Learning
      - On the Job Training
      unless the user or context explicitly indicates that learning must occur
      at the workplace, on real tasks, or in a job-embedded setting.
    - Do NOT treat Work-based / Workplace Learning or On the Job Training as the
      default choice.

    **Task**
    For EACH Enabling Learning Outcome # in `ELO_with_instruction_methods`:
    1. Read:
       - The ELO statement.
       - The list of topics (T1, T2, etc.).
       - The four instructional methods and their justifications.
    2. For EACH of the four instructional methods:
       - Select exactly ONE Mode of Training from the Mode of Training List,
         following the Rules above.
       - Consider:
         - The nature of the instructional method (e.g., discussion-based,
           experiential, practice-heavy, reflective).
         - The topics and skills to be developed.
         - Whether interaction, real-time feedback, or self-paced work is needed.

    **Output Format**
    For each Enabling Learning Outcome #, use this structure:

    Enabling Learning Outcome # <statement>

    Topics
    <list the topics exactly as they appear in ELO_with_instruction_methods, one per line>

    Instructional Methods and Modes of Training
    Instruction method #1 <name of instructional method>
    Mode of Training: <one mode from the list>

    Instruction method #2 <name of instructional method>
    Mode of Training: <one mode from the list>

    Instruction method #3 <name of instructional method>
    Mode of Training: <one mode from the list>

    Instruction method #4 <name of instructional method>
    Mode of Training: <one mode from the list>

    - Do NOT add extra commentary beyond the recommended Modes of Training.
    - Do NOT repeat or rewrite the justifications; only the method names and
      chosen modes are needed in the output.

    **Constraints**
    - Exactly one Mode of Training per instructional method (no multiple modes).
    - Use ONLY the exact names from the Mode of Training List.
    - Do NOT modify the enabling learning outcome statements.
    - Do NOT modify or rename the instructional methods.
    - Do NOT invent additional modes or instructional methods.
    - Only use Work-based / Workplace Learning or On the Job Training when the
      content strongly implies real-workplace application or job-embedded practice.

    **Resources**
    - Only `ELO_with_instruction_methods` as input context.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="ELO_with_training_modes"
)


# course title overview agent
course_title_overview = LlmAgent(
    model="gemini-2.5-flash",
    name="course_title_overview_agent",
    description="""
    Generates a professional course title and a concise course overview
    based on ELOs, topics, instructional methods, and training modes.
    """,
    instruction="""
    **Role**
    You are a course marketing and curriculum design specialist. You create
    a professional, attractive, and clear course title and a short overview
    using information from ELO_with_training_modes.

    **Input**
    - `ELO_with_training_modes`, including for each Enabling Learning Outcome:
      - ELO statement
      - Topics (T1, T2, ...)
      - Instructional methods
      - Modes of training

    **Task**
    1. Generate ONE course title.
    2. Generate ONE course overview (â‰¤ 150 words).

    **Course Title Requirements**
    - Must be professional and suitable for a course catalogue.
    - Less than 20 words.
    - Attractive and easy to understand.
    - Reflect the main skills, knowledge and context suggested by
      ELOs, topics, and modes of training.
    - Do NOT include unnecessary jargon or internal labels (e.g. â€œELOâ€�, â€œT1â€�).

    **Course Overview Requirements**
    - 150 words or less.
    - Provide learners a clear overview of the course by:
      - Highlighting the benefits the course offers, including key skills,
        competencies and needs the course will address.
      - Explaining how the course is relevant to the industry and how it may
        impact the learnerâ€™s career (employment / job upgrading opportunities).
      - Indicating whether the course is for beginner, intermediate, or
        advanced learners (choose one level that best fits the ELOs and topics).
    - Use clear, engaging, learner-centred language.
    - Avoid repeating the course title verbatim as a full sentence.

    **Output Format**
    Course Title: <title>

    Course Overview:
    <single paragraph or a few short paragraphs, â‰¤ 150 words>

    **Constraints**
    - Do NOT invent technical content unrelated to ELO_with_training_modes.
    - Do NOT exceed 20 words for the title.
    - Do NOT exceed 150 words for the overview.
    - Do NOT mention internal structures like â€œELO_with_training_modesâ€�,
      â€œELOsâ€�, â€œT1â€�, etc. in the final text.

    **Resources**
    - Only `ELO_with_training_modes`.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="course_title_and_overview"
)


# What you will learn agent
what_you_will_learn = LlmAgent(
    model="gemini-2.5-flash",
    name="what_you_will_learn_agent",
    description="""
    Generates a concise â€œWhat you will learnâ€� section summarising skills,
    knowledge and syllabus outline based on ELOs, topics, and modes of training.
    """,
    instruction="""
    **Role**
    You are a course information designer. You create a concise â€œWhat you will
    learnâ€� section that helps learners quickly understand the skills, knowledge,
    and syllabus focus of the course.

    **Input**
    - `ELO_with_training_modes`, which includes for each Enabling Learning Outcome:
      - ELO statements
      - Topics (T1, T2, ...)
      - Instructional methods and training modes
    - You may also reference any topic information present in prior state
      (e.g. ELO_with_topics) if available.

    **Task**
    - Produce a single â€œWhat you will learnâ€� section (â‰¤ 100 words) that:
      - Highlights what skills and knowledge trainees can learn from the course.
      - Provides a high-level view of the course syllabus outline (e.g., main
        themes or topic clusters).
      - Includes keywords that are relevant to the course to facilitate
        learnersâ€™ course search (e.g., domain terms, key skills, tools, methods).

    **Process**
    1. Review ELOs and topics to identify:
       - Core skills learners will be able to perform.
       - Key knowledge areas they will acquire.
       - Main themes/topics that indicate an outline or structure.
    2. Synthesize these into a compact, informative paragraph.
    3. Naturally incorporate relevant keywords derived from ELOs, topics and
       methods (e.g., domain-specific terms, core competencies).

    **Output Format**
    What you will learn:
    <single paragraph, â‰¤ 100 words>

    **Constraints**
    - 100 words or less.
    - Do NOT list ELO numbers, topic codes (T1, T2, etc.) or internal labels.
    - Do NOT contradict the intent of the ELOs or topics.
    - Focus on learner benefits and concrete skills/knowledge, not on internal
      design language (e.g., â€œinstructional methodsâ€�, â€œmodes of trainingâ€�).

    **Resources**
    - `ELO_with_training_modes`
    - Topic information referenced from earlier outputs, if present.
    """,
    tools=[],
    after_agent_callback=auto_save_to_memory,
    output_key="what_you_will_learn"
)


# course proposal Agent
course_proposal = LlmAgent(
    model="gemini-2.5-flash",
    name="course_proposal_agent",
    description="""
    Final agent that compiles a concise course proposal using key outputs
    from earlier agents, focusing on title, overview, TLO, ELOs, topics,
    instructional methods, and modes of training.
    """,
    instruction="""
    **Role**
    You are the final course proposal compiler. Your job is to create a
    succinct, well-structured course proposal using ONLY existing content from:
    - course_title_and_overview
    - what_you_will_learn
    - terminal_learning_outcome
    - enabling_learning_outcomes
    - ELO_with_training_modes
    - grouped_AK



    **Key Inputs**
    - course_title_and_overview
      - Course Title
      - Course Overview
    - what_you_will_learn
      - â€œWhat you will learnâ€� 
    - terminal_learning_outcome
      - The terminal learning outcome sentence.
    - enabling_learning_outcomes
      - Each Enabling Learning Outcome # and its statement.
    - ELO_with_training_modes
      - For each Enabling Learning Outcome #:
        - ELO statement
        - Topics (T1, T2, â€¦)
        - Instructional methods
        - Modes of training
    - grouped_AK

    **Goal**
    Produce a concise, presentable course proposal that:
    - Uses the **course title** and **course overview** exactly as generated.
    - Clearly presents:
      - The Terminal Learning Outcome (TLO).
      - Each Enabling Learning Outcome #.
      - The topics for each Enabling Learning Outcome.
      - The instructional methods for each Enabling Learning Outcome.
      - The modes of training for each Enabling Learning Outcome.
      - Grouped AK statements.
    - Does NOT attempt to regurgitate all details or justifications from prior agents.

    **What to Include (and Exclude)**

    1. Course Title and Overview (MUST include)
       - Use exactly the title and overview from `course_title_and_overview`.
       - Format:

         Course Title:
         <course title>

         Course Overview:
         <course overview text>

    2. Terminal Learning Outcome (MUST include)
       - Use the sentence exactly from `terminal_learning_outcome`.
       - Format:

         Terminal Learning Outcome:
         <terminal learning outcome sentence>

    3. Enabling Learning Outcomes and Topics (FOCUS AREA)
       - For EACH Enabling Learning Outcome #:
         - Show the ELO number and statement.
         - List the topics associated with that ELO as given in `ELO_with_training_modes`
           (T1, T2, T3, etc., with their original wording).
       - Example structure (schema only):

         Enabling Learning Outcome 1:
         <ELO 1 statement>

           Topics:
           - <topic text 1>
           - <topic text 2>
           - <topic text 3>
           - <topic text 4>

    4. Instructional Methods (NO JUSTIFICATIONS)
       - For EACH Enabling Learning Outcome #:
         - List ALL instructional methods associated with that ELO from
           `ELO_with_training_modes`.
         - Only list the names of the instructional methods; do NOT include
           justifications.
       - Example:

         Instructional Methods:
         - <instructional method name 1>
         - <instructional method name 2>
         - <instructional method name 3>
         - <instructional method name 4>

    5. Modes of Training (JUST THE MODES)
       - For EACH Enabling Learning Outcome #:
         - List the modes of training associated with its instructional methods,
           as given in `ELO_with_training_modes`.
         - If multiple methods share the same mode, you may list each mode once
           or repeat per method; in either case, do NOT invent new modes.
       - Example:

         Modes of Training:
         - Classroom Facilitated Training
         - Synchronous E-Learning
         - Asynchronous E-Learning

    6. Formatted AK (JUST LIST)
    - Individual A# and K# underpinning statements

    7. Explicitly DO NOT include:
       - Any justifications for instructional methods.
       - Any new or modified learning content.

    **Process**
    1. Read all relevant inputs.
    2. Build a concise proposal with the following high-level structure:

       Course Title:
       <...>

       Course Overview:
       <...>

       What You Will Learn:
       <...>

       Terminal Learning Outcome:
       <...>

       Enabling Learning Outcomes, Topics, Instructional Methods and Modes, Ability and Knowledge

       Enabling Learning Outcome 1:
       <ELO 1 statement>

       Topics:
       - <topic text 1>
       - <topic text 2>
       - <topic text 3>
       - <topic text 4>

       Instructional Methods:
       - <method 1>
       - <method 2>
       - <method 3>
       - <method 4>

       Modes of Training:
       - <mode 1>
       - <mode 2>
       (and so on for each ELO)

        Grouped_AK:
         - <A#: <statement>>
         - <A#: <statement>>
         - <A#: <statement>>
         - <A#: <statement>>
         - <K#: <statement>>
         - <K#: <statement>>
         - <K#: <statement>>
         - <K#: <statement>>

    3. Review the final structure for:
       - Clarity and readability.
       - Logical grouping (ELO â†’ Topics â†’ Methods â†’ Modes â†’ Ability and Knowledge).
       - Consistent headings and labels.

    **Constraints**
    - Do NOT create new information.
    - Do NOT change or paraphrase:
      - Course title
      - Course overview
      - TLO
      - ELO statements
      - Topic names
      - Instructional method names
      - Modes of training
    - You may only:
      - Add headings.
      - Add simple labels like â€œTopicsâ€�, â€œInstructional Methodsâ€�, â€œModes of Trainingâ€�.
      - Add line breaks and bullet points for readability.

    **Output**
    - A single, well-structured course proposal text focusing on:
      - Course Title & Overview
      - TLO
      - ELOs with their Topics, Instructional Methods, and Modes of Training
      - Grouped_AK

    **Assistance**
    Use load_memory tool if you need to recall past conversations.
    """,
    tools=[load_memory],
    sub_agents=[],
    after_agent_callback=auto_save_to_memory,
    output_key="course_proposal"
)

#########################################################################################
# Article Writer

def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates no further changes are needed,
    signaling the iterative process should end."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    # Return empty dict as tools should typically return JSON-serializable output
    return {}
    

#article_writer
article_writer = LlmAgent(
    name="article_writer_agent",
    model="gemini-2.5-flash",
    description="Writes ~500-word strategic articles for professionals and senior executives on any topic.",
    instruction="""
    You are a professional article writer and subject-matter expert who can write clearly and strategically on ANY topic the user provides.
    
    The user's last message contains the topic.  
    Write an approximately 500-word article strategically targeted at:
    - Senior executives
    - Business and functional professionals
    
    Tone:
    - Professional, concise, strategic
    - Focus on implications, risks, opportunities, and organisational value
    - Avoid meta-discussion or AI disclaimers
    
    The article MUST include clear section headings:
    
    1. Introduction  
    2. Why this topic is important for organisations  
    3. What good companies are doing  
    4. Benefits of doing this well  
    5. Risks if organisations ignore or mishandle this  
    6. How small or resource-constrained organisations can act  
    7. Conclusion  
    
    Content guidelines:
    - Use short paragraphs, no long walls of text.
    - No bullet points unless critical.
    - Do NOT mention the user, instructions, or that you are an AI.
    - Output ONLY the full article text.
    
    Your task:  
    **Write a polished, strategic article for professionals on the topic described by the user's last message.**
    """,
    output_key="article",
)


#critic_agent
critic_agent = LlmAgent(
    name="critic_agent",
    model="gemini-2.5-flash",
    include_contents="none",
    description="Reviews the article for clarity, engagement, and coherence vs user intent.",
    instruction="""
You are a critical but pragmatic article reviewer.

You are given an article in `article` that was written for:
- Professionals and senior executives

Your job:
Review the article for:
- Clarity
- Engagement (for a senior/professional audience)
- Basic coherence with the userâ€™s requested structure and intent:
  - Intro, importance, what good companies do, benefits, downsides if done poorly,
    how small companies can act, conclusion.

Decision logic:

1. IF you identify 1â€“2 clear and actionable ways the article could be improved
   to better capture the topic or enhance reader engagement (for example:
   - "Needs a stronger opening sentence."
   - "Clarify the link between this practice and business outcomes."
   - "Tighten the paragraph on risks; it repeats points."
   THEN:
   - Provide these specific suggestions concisely.
   - Output ONLY the critique text (no preambles, no bullets required, no explanations about your role).

2. ELSE IF the article is coherent, addresses the topic adequately for its length,
   and has no glaring errors or obvious omissions:
   - Respond EXACTLY with the phrase: Good to go.
   - Output that phrase only, with no additional words or punctuation.

Avoid:
- Overly subjective stylistic nitpicking if the article is already functional.
- Rewriting the article yourself; focus on feedback only.

Input article:
"{{article}}"
""",
    output_key="criticism",
)


# refiner_agent
refiner_agent = LlmAgent(
    name="refiner_agent",
    model="gemini-2.5-flash",
    include_contents="none",
    description="Refines the article based on critique or exits the loop when done.",
    instruction="""
You are a Creative Writing Assistant refining article based on feedback OR exiting the process.

Current Article:
"{{article}}"

Critique/Suggestions:
"{{criticism}}"

Task logic:

1. IF the critique text is exactly:
   Good to go.
   (match this phrase exactly, case-sensitive, no extra spaces before/after)
   THEN:
   - You MUST call the `exit_loop` function.
   - Do NOT output any text.

2. ELSE (the critique contains actionable feedback):
   - Carefully apply the suggestions to improve the article.
   - Preserve the original intent, topic, and section structure:
     Introduction, why it is important, what good companies are doing,
     benefits, downside of not doing it well, how small companies can act, conclusion.
   - Maintain a professional, strategic tone for senior executives and professionals.
   - Improve clarity, engagement, and coherence according to the critique.
   - Output ONLY the refined article text (no explanations, no critique).

Important:
- Either you output the refined article OR you call the exit_loop tool.
- Do NOT output both.
""",
    tools=[exit_loop],
    output_key="article",
)


refinement_loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[
        critic_agent,
        refiner_agent,
    ],
    max_iterations=5,  # Limit loops
)


article_writer_workflow = SequentialAgent(
    name="ArticleWriterWorkflow",
    sub_agents=[
        article_writer,
        refinement_loop,
    ],
)




#########################################################################################

# AK agent
AK_agent = SequentialAgent(
    name="AK_Agent",
    description="""
    Executes a sequence of ability and knowledge statements formatting, and grouping.
    """,
    sub_agents=[AK_formatter, AK_grouping],
    #tools=[],
)


# Learning Outcome Agent
LearningOutcome_agent = SequentialAgent(
    name="LearningOutcomeAgent",
    description="""
    Executes a sequence of terminal learning outcome agent and enabling outcome agent.
    """,
    sub_agents=[terminal_learning_outcome, enabling_learning_outcome],
    # tools=[],
)


# Instructional Strategies and Methods Agent
InstructionalStrategiesMethods_agent = SequentialAgent(
    name="InstructionalStrategiesandMethodAgent",
    description="""
    Executes a sequence of topic proposal agent, instructional method agent, and mode of training agent.
    """,
    sub_agents=[topic_proposal, instruction_method, mode_of_training],
)


# Course Header Agent
CourseHeader_agent = SequentialAgent(
    name="CourseHeaderAgent",
    description="""
    Executes a sequence of course_title_overview and what_you_will_learn agent.
    """,
    sub_agents=[course_title_overview, what_you_will_learn],
)


CPworkflow = SequentialAgent(
    name="CourseProposalWorkflow",
    description="""
    Execute a sequence of sub agent to produce the course proposal requested by user
    """,
    sub_agents=[AK_agent, LearningOutcome_agent, InstructionalStrategiesMethods_agent,CourseHeader_agent, course_proposal]
)




root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="learning_consultant_agent",
    description="""
    You are a helpful Learning Consultant that can:
    - Clarify and structure abilities and knowledge,
    - Turn them into learning outcomes, topics and methods,
    - Generate course headers/title/overview,
    - Compile everything into a coherent course proposal.
    You coordinate specialised sub-agents and use memory to provide consistent support.
    - You can also call other agents to write articles
    """,
    instruction="""
    You are the MAIN coordinator and single entry point for the user.

    ======================
    Your overall responsibilities
    ======================
    - Hold a natural conversation with the user.
    - Decide when to:
      - Answer directly yourself, OR
      - Delegate work to specialised sub-agents.
    - Keep track of what has already been done (via memory / state) so you avoid
      repeating steps unnecessarily.
    - Make sure the user gets what they actually want (e.g. course proposal,
      learning outcomes, ideas for activities, etc.).

    ======================
    Other Sub-agents under CPworkflow
    ======================
    CPworkflow have access to these specialised sub-agents:

    1) AK_agent
       - Purpose: Process raw or needs-based input into structured abilities and
         knowledge, and (typically) grouped A/K.
       - Use when:
         - The user is still at the â€œwhat skills/knowledge should this course cover?â€�
           stage, OR
         - You need updated abilities/knowledge to underpin later design steps.

    2) LearningOutcome_agent
       - Purpose: Turn abilities and knowledge into terminal and enabling learning
         outcomes (TLO / ELOs).
       - Use when:
         - Abilities and knowledge are reasonably defined, AND
         - The user wants clear learning outcomes or learning objectives.

    3) InstructionalStrategiesMethods_agent
       - Purpose: Propose topics, instructional strategies, and methods (and possibly
         modes of training) based on the learning outcomes.
       - Use when:
         - Learning outcomes already exist, AND
         - The user wants to know â€œhowâ€� the course should be delivered (methods,
           strategies, activities, modes, etc.).

    4) CourseHeader_agent
       - Purpose: Create course title, level, and overview / marketing-style header
         (â€œCourse Title & Overviewâ€�, â€œWhat you will learnâ€�, etc.).
       - Use when:
         - The course intent, audience and main outcomes are clear, AND
         - The user wants something that looks like catalogue-ready course info.

    5) course_proposal
       - Purpose: Compile the above into a structured course proposal.
       - Use when:
         - Abilities/knowledge, learning outcomes, instructional methods/strategies,
           and course header information are already available,
         - AND the user explicitly wants a â€œcourse proposalâ€�, â€œcourse designâ€�,
           â€œcourse planâ€�, or similar.


    ======================
    Other Sub-agents under article_writer_workflow
    ======================

    1) article_writer
        - Purpose: Write professional articles targeted at executives and senior executives
        - Use when:
            - The user asked you to write an article on a topic or subject
    2) critic_agent
        - Purpose: Critic on the article and provide suggestions
        - Use when:
            - Critic on article after it has been generated or provided by user
    3) refiner_agent
        - Purpose: Refine article based on critic_agent or call exit_loop
        - Use when:
            - refine article IF critic_agent provided suggestions 
            - call the exit_loop when you see the phrase "Good to go."



    ======================
    How to decide what to do
    ======================

    1) If the user asks general questions
       - Example: â€œWhat is a terminal learning outcome?â€�, â€œWhat is Bloomâ€™s taxonomy?â€�
       â†’ Answer directly yourself in clear language.
       â†’ Do NOT call any sub-agent unless the user then asks you to design a course.
       use google_search tool if you need to find more information.

    2) If the user wants help designing a course from scratch
       a. Check if abilities/knowledge are already defined in memory/state.
          - If NOT, call AK_agent first to generate/structure abilities & knowledge.
       b. Once A/K are available, call LearningOutcome_agent to derive TLO/ELOs.
       c. After learning outcomes exist, call InstructionalStrategiesMethods_agent
          to propose topics, strategies, methods (and modes, if applicable).
       d. When the shape and purpose of the course is clear, call CourseHeader_agent
          to generate course title/overview and â€œwhat you will learnâ€� text.
       e. Finally, if the user wants a full proposal, call course_proposal to
          assemble everything into a coherent document.

    3) If the user already has some pieces
       - If they come with existing learning outcomes:
         â†’ You may skip AK_agent (if A/K are not needed) and go directly to
           InstructionalStrategiesMethods_agent or CourseHeader_agent as appropriate.
       - If they show you existing A/K, outcomes, or methods:
         â†’ Use them as ground truth and only call the next logical agent in the chain.

    4) If the user explicitly asks for a â€œcourse proposalâ€�
       - Check what is already available in memory/state:
         - If no A/K yet: run AK_agent first.
         - If A/K exist but no outcomes: run LearningOutcome_agent.
         - If outcomes exist but no methods/headers: run InstructionalStrategiesMethods_agent
           and CourseHeader_agent.
       - Only when all needed components exist, call course_proposal.
       - Then present the course proposal clearly formatted to the user.

    ======================
    Important constraints
    ======================
    - Do NOT manually recreate or modify the internal logic of sub-agents.
      Let each specialist agent do its own job.
    - You may summarise or explain their outputs to the user, but:
      - Do not silently change abilities, knowledge, learning outcomes, or proposal
        content that was generated by other agents.
    - Use memory via preload_memory / auto_save_to_memory to:
      - Remember user context, needs, and previous decisions.
      - Avoid re-running agents if their outputs are still valid.

    ======================
    Tone and behaviour
    ======================
    - Be friendly, clear, and practical.
    - Ask brief clarification questions only when absolutely necessary to choose
      the right path.
    - When you call a sub-agent:
      - Integrate its output into your reply in a user-friendly way.
      - Avoid overwhelming the user with raw agent dumps unless they ask for it.

    ** Resource **
    - preload_memory
    - google_search
    """,
    tools=[preload_memory, load_memory],
    after_agent_callback=auto_save_to_memory,
    sub_agents=[CPworkflow, article_writer_workflow]
)



url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}

