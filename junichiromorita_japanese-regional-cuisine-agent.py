# =========================================
# 0. Setup - Install required packages
# =========================================
!pip install -q google-adk google-genai datasets pandas


# =========================================
# 1. Imports & logging
# =========================================
# This cell wires up Gemini + ADK and basic logging.
# Logging is used later in the multi-turn example to demonstrate observability.
import os
import logging
import warnings
import re

import pandas as pd
from IPython.display import Markdown, display

from google.genai import types as genai_types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is empty.")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… GOOGLE_API_KEY loaded successfully from Kaggle Secrets.")
except Exception as e:
    print("âš ï¸� Could not load GOOGLE_API_KEY from Kaggle Secrets:", e)
    print("   If you run this notebook outside Kaggle, please set os.environ['GOOGLE_API_KEY'] manually.")

# Create a dedicated logger for this notebook
logger = logging.getLogger("japanese-regional-cuisine-agent")
logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers when re-running the cell
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("Logging initialized (stream only).")

# Silence function_call warning from google-genai types module
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

print("âœ… Imports, logging, and warning suppression are configured.")


# =========================================
# 2. Load Our-Regional-Cuisines (English)
# =========================================
ENG_CSV_URL = (
    "https://huggingface.co/datasets/JunichiroMorita/Our-Regional-Cuisines/"
    "raw/main/our_regional_cuisines_eng.csv"
)

df = pd.read_csv(ENG_CSV_URL)

print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))
df.head(1)


# =========================================
# 3. Base agent factory (ADK)
# =========================================
# This helper centralizes how we construct LlmAgent instances:
# - model configuration (Gemini 2.5 Flash)
# - temperature / max tokens
# - optional tools and output key for ADK state integration.

def create_base_agent(
    name: str,
    instruction: str,
    tools: list | None = None,
    output_key: str | None = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.5,
    max_output_tokens: int = 1024,
) -> LlmAgent:
    """
    Helper to create a base LlmAgent.

    Note:
    - Session and memory are managed by the Runner (InMemoryRunner),
      so they must NOT be passed into LlmAgent here.
    """
    generate_content_config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    agent = LlmAgent(
        model=model_name,
        name=name,
        instruction=instruction,
        tools=tools or [],
        output_key=output_key,
        generate_content_config=generate_content_config,
    )

    logger.info("Created agent: %s", name)
    return agent


print("âœ… Base agent factory is defined.")


# =========================================
# 4. Parsing helper and structured entry builder
# =========================================
# This cell converts each raw "text" row from the dataset into
# a structured dictionary (name, region, ingredients, history, etc.)
# so that tools and the agent can reason over explicit fields.

def extract_field(pattern: str, text: str) -> str:
    """
    Extract a single field from the raw text using a regex pattern
    with one capturing group. Returns an empty string if not found.
    """
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def parse_entry_basic(text: str) -> dict:
    """
    Parse the raw 'text' of one cuisine entry into a basic structured dict.
    The patterns are based on the Our Regional Cuisines markdown-like format.
    """
    cuisine_name = extract_field(
        r"\*\*Cuisine Name\*\*:\s*(.+?)\s*\*\*Region\*\*:",
        text,
    )

    region = extract_field(
        r"\*\*Region\*\*:\s*(.+?)\s*##\s*Main Lore Areas",
        text,
    )

    main_lore_areas = extract_field(
        r"##\s*Main Lore Areas\s*(.+?)\s*##\s*Main Ingredients Used",
        text,
    )

    main_ingredients = extract_field(
        r"##\s*Main Ingredients Used\s*(.+?)\s*##\s*History, Origin, and Related Events",
        text,
    )

    history = extract_field(
        r"##\s*History, Origin, and Related Events\s*(.+?)\s*##\s*Opportunities and Times of Eating Habits",
        text,
    )

    opportunities = extract_field(
        r"##\s*Opportunities and Times of Eating Habits\s*(.+?)\s*##\s*How to Eat",
        text,
    )

    how_to_eat = extract_field(
        r"##\s*How to Eat\s*(.+?)\s*##\s*Efforts for Preservation and Succession",
        text,
    )

    ingredients_section = extract_field(
        r"##\s*Ingredients\s*(.+?)\s*##\s*Recipe",
        text,
    )

    recipe_section = extract_field(
        r"##\s*Recipe\s*(.+?)\s*##\s*Provider Information",
        text,
    )

    provider = extract_field(
        r"##\s*Provider Information\s*(.+)$",
        text,
    )

    return {
        "cuisine_name": cuisine_name,
        "region": region,
        "main_lore_areas": main_lore_areas,
        "main_ingredients": main_ingredients,
        "history": history,
        "opportunities": opportunities,
        "how_to_eat": how_to_eat,
        "ingredients_section": ingredients_section,
        "recipe_section": recipe_section,
        "provider": provider,
        "raw_text": text,
    }


print("âœ… Parsing helpers are defined.")


# =========================================
# 5. Search tools on top of the single 'text' column
# =========================================
# These functions implement the "tool layer" for the agent.
# ADK exposes them as tools so the LlmAgent can call:
#   - search_by_region()
#   - search_by_dish_name()
#   - search_by_ingredient()
# over the curated Hugging Face dataset.

def search_by_region(region: str, limit: int = 5) -> list:
    """
    Search cuisines whose 'Main Lore Areas' section mentions the given region.
    Returns a list of parsed cuisine dicts.
    """
    region_lower = region.lower()
    results = []

    for text in df["text"]:
        main_lore_areas = extract_field(
            r"##\s*Main Lore Areas\s*(.+?)\s*##\s*Main Ingredients Used",
            text,
        )
        if main_lore_areas and region_lower in main_lore_areas.lower():
            results.append(parse_entry_basic(text))
            if len(results) >= limit:
                break

    return results


def search_by_dish_name(name: str, limit: int = 5) -> list:
    """
    Search cuisines whose 'Cuisine Name' matches or partially matches the given name.
    """
    name_lower = name.lower()
    results = []

    for text in df["text"]:
        cuisine_name = extract_field(
            r"\*\*Cuisine Name\*\*:\s*(.+?)\s*\*\*Region\*\*:",
            text,
        )
        if cuisine_name and name_lower in cuisine_name.lower():
            entry = parse_entry_basic(text)
            results.append(entry)
            if len(results) >= limit:
                break

    return results


def search_by_ingredient(ingredient: str, limit: int = 5) -> list:
    """
    Search cuisines whose 'Main Ingredients Used' section contains the given ingredient.
    """
    ingredient_lower = ingredient.lower()
    results = []

    for text in df["text"]:
        main_ingredients = extract_field(
            r"##\s*Main Ingredients Used\s*(.+?)\s*##\s*History, Origin, and Related Events",
            text,
        )
        if main_ingredients and ingredient_lower in main_ingredients.lower():
            results.append(parse_entry_basic(text))
            if len(results) >= limit:
                break

    return results

print("âœ… Search functions (tools) are defined.")


# =========================================
# 6. Markdown formatter for a cuisine entry
# =========================================
# This formatter turns a structured cuisine dict into a
# readable Markdown block, used in the agent's final answers.

def format_cuisine_markdown(entry: dict) -> str:
    """
    Convert a parsed cuisine entry (dict) into a clean, readable Markdown block.
    """
    name = entry.get("cuisine_name") or "(Unknown Cuisine Name)"
    region = entry.get("region") or "(Unknown region)"
    ingredients = entry.get("main_ingredients") or ""
    history = entry.get("history") or ""
    how = entry.get("how_to_eat") or ""
    recipe = entry.get("recipe_section") or ""

    md = f"""
## ğŸ�½ï¸� {name}

**Region:** {region}  
**Main Ingredients:** {ingredients}

### ğŸ“� Overview
{history if history else "No detailed history available in this dataset."}

### ğŸ�´ How It Is Enjoyed
{how if how else "No specific notes on how it is eaten."}

### ğŸ‘¨â€�ğŸ�³ Recipe Notes
{recipe if recipe else "No recipe details available in this dataset."}
""".strip()

    return md


print("âœ… Markdown formatter is defined.")


# =========================================
# 7. System prompt for UserGuideAgent (multi-turn + readable output)
# =========================================
# This system prompt encodes one of the core design innovations of this agent:
# A lightweight "cultural & religious dietary constraint engine" built directly
# into the instruction layer. It defines how the agent evaluates tool results
# (e.g., Halal, Kosher, Hindu, Buddhist vegetarian, Jain restrictions) and
# filters out unsafe or inappropriate dishes before generating the final answer.
#
# This results in behavior that a simple LLM prompt cannot achieve:
# - strict rule-based filtering of search candidates,
# - safety-aware recommendations,
# - consistent reasoning across multi-turn sessions.
# This makes the system more trustworthy and more suitable for international travelers.

USER_GUIDE_SYSTEM_PROMPT = """
You are the â€œJapanese Regional Cuisine User Guide Agent,â€� a friendly and culturally aware assistant
who helps travelers discover traditional Japanese dishes.

Speak naturally, like a local who loves talking about food.
Keep explanations warm and conversational, not like a textbook.

================================
1) Understand the user first
================================
From the userâ€™s words, carefully pick up:
- Preferences: light / rich, vegetable-based, seafood, etc.
- Dietary rules: vegetarian, vegan, halal, kosher, Hindu (no beef), etc.
- Explicit â€œnoâ€� items: â€œno porkâ€�, â€œno meatâ€�, â€œno alcoholâ€�, allergies.
- Context: visiting Japan, eating out, or cooking at home after returning.

If something important is unclear (for example, â€œIs fish OK for you?â€�),
ask ONE short follow-up question. Do not bombard the user with many questions.

================================
2) How to use tools and filter candidates
================================
You have access to tools that search the dataset:
- search_by_region(region: str)
- search_by_dish_name(name: str)
- search_by_ingredient(ingredient: str)

Use them when you need concrete candidates.
However:
- Do NOT show or mention the tools to the user.
- The user should only see your final, human-like answer.

When you receive a list of candidate dishes from tools, you MUST:
1. Read the userâ€™s request again.
2. For each candidate dish, check its fields (cuisine_name, main_ingredients, raw_text).
3. Decide if the dish really fits the userâ€™s preferences and restrictions.

If a dish clearly conflicts (for example):
- The user wants â€œlight, vegetable-basedâ€� and the dish is meat-heavy.
- The user is Muslim and the dish uses pork or alcohol-based seasoning.
- The user is Hindu and the dish uses beef.
then you MUST NOT recommend that dish.

If you are not sure whether a dish is truly suitable:
- Be honest about the uncertainty.
- Prefer safer alternatives that clearly match the userâ€™s diet.

================================
3) Religious and cultural dietary care
================================
Be especially careful around religious dietary rules, such as:
- Halal (Islam): avoid pork, avoid alcohol in cooking (mirin, sake, etc.),
  and be cautious with non-halal meat.
- Kosher (Judaism): avoid pork and shellfish; avoid mixing meat and dairy.
- Hindu dietary culture: avoid beef; some people avoid all meat.
- Buddhist vegetarian / vegan: avoid meat and fish; some avoid strong-smelling plants
  like onion and garlic.
- Jain: strict vegetarian, no root vegetables (onion, garlic, potatoes, carrots, etc.).

When you suspect such rules:
- Treat them as strict constraints unless the user clearly relaxes them.
- It is better to say â€œthis may not fully meet your religious rulesâ€� than to overpromise.
- You can gently suggest safer plant-based or simple dishes when in doubt.

You may draw on general ideas from Japanese multicultural food guidance
(e.g. Japanâ€™s tourism manuals about serving foreign guests),
but do not claim legal or religious authority.

================================
4) How to present recommendations
================================
Choose 1â€“3 dishes that best match the user.
Then answer in a single, smooth response, not as a list of tool results.

When the userâ€™s dietary rules greatly limit the available dishes in a region, 
you should expand your explanation by adding:

- seasonal cultural context,
- temple-food traditions (e.g., Shojin-ryori),
- ingredient stories (e.g., Japanese local vegetables),
- ways Halal travelers can still enjoy local flavors,
- specific types of restaurants where suitable dishes are found.

Do NOT shorten your answer. Provide at least two culturally rich details 
so the user learns something memorable about the region's cuisine.

You may use light Markdown to make it readable, for example:

## ğŸ�½ï¸� <Dish name> â€” <Region>
A short, friendly description (2â€“3 sentences) about what this dish is like
and why it might suit the userâ€™s taste or diet.

- Main ingredients: ...
- When it is eaten: festival / home / winter, etc.
- Any important notes: for example, â€œUses dashi made from fish,â€�
  or â€œUsually does not contain meat, so itâ€™s often suitable for vegetarians.â€�

Explain things in a way that feels like a local recommending their favorite foods,
not like a database printout.

================================
5) Eating out and cooking at home
================================
If the user is looking for places to eat:
- Describe the types of restaurants or shops where the dish is usually served
  (for example, izakaya, local diners, specialty tofu restaurants).
- Do NOT invent specific shop names or addresses unless a trusted search tool provides them.

If the user wants to recreate the dish at home after returning to their country:
- Ask (or use known info) about which country or region they live in.
- Suggest realistic ingredient substitutions available there.
- Explain how the substitution changes the flavor or texture.
- Try to preserve the â€œspiritâ€� and cultural meaning of the original dish.

================================
6) Tone and closing
================================
- Use clear, natural English, with a warm and relaxed tone.
- Itâ€™s okay to say things like â€œOh, Kyoto is a fantastic choice for light dishes.â€�
- Avoid very stiff or overly formal sentences.
- At the end, ask a gentle follow-up such as:
  â€œWould you like to hear more about one of these dishes, or explore another region?â€�
- You may add a short Japanese phrase like ã€Œç¾�å‘³ã�—ã�§ã�™ã‚ˆã€� or ã€Œã�œã�²ã�Šè©¦ã�—ã��ã� ã�•ã�„ã€�.
""".strip()

# Create UserGuideAgent with tools
user_guide_agent = create_base_agent(
    name="UserGuideAgent",
    instruction=USER_GUIDE_SYSTEM_PROMPT,
    tools=[search_by_region, search_by_dish_name, search_by_ingredient],
    output_key=None,
)

runner = InMemoryRunner(
    agent=user_guide_agent,
    app_name="Japanese-Regional-Cuisine-Agent",
)

print("âœ… UserGuideAgent (gemini-2.5-flash, natural style) is ready.")


# =========================================
# 8. Helper to extract final text from debug events
# =========================================
# run_debug() returns a sequence of ADK events.
# This helper walks through them to collect the final LLM text answer
# so we can render it cleanly in the notebook demos.

from typing import List

def extract_final_text(events: List) -> str:
    """
    Extract the final text answer from ADK debug events returned by run_debug().
    """
    final_text = None
    for e in events:
        if hasattr(e, "content") and e.content:
            for part in e.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text = part.text
    return final_text or ""


# =========================================
# 9. Conversation test with UserGuideAgent
# =========================================

# query = "I like seafood and I am curious about dishes from Hokkaido. What would you recommend?"

# events = await runner.run_debug(query)
# answer_text = extract_final_text(events)

# print("=== Raw text (for debug) ===")
# print(answer_text[:300] + "..." if len(answer_text) > 300 else answer_text)

# print("\n=== Rendered Markdown ===")
# display(Markdown(answer_text))


# =========================================
# 10. Helper: run a query and display conversation nicely
# =========================================
# This convenience function:
# - runs UserGuideAgent via runner.run_debug()
# - suppresses internal debug prints
# - renders the user message and agent reply as clean Markdown
#   so that the notebook can showcase realistic conversations.

import io
import contextlib

async def chat_and_show(user_message: str):
    """
    Run UserGuideAgent with the given user message and
    display the conversation in a user-friendly Markdown format.
    Internally uses runner.run_debug(), but suppresses its prints.
    """
    # Capture and suppress stdout from run_debug (which prints "User > ..." etc.)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        events = await runner.run_debug(user_message)

    # Extract final text from events
    answer_text = extract_final_text(events)

    # User message (quoted)
    display(Markdown(f"### ğŸ§‘ User\n\n> {user_message}"))

    # Agent response (Markdown as-is)
    display(Markdown(f"### ğŸ¤– UserGuideAgent\n\n{answer_text}"))

# await chat_and_show("I like seafood and I am curious about dishes from Hokkaido. What would you recommend?")


await chat_and_show(
    "I love seafood and I'm interested in traditional dishes from Hokkaido. "
    "I'm visiting in winterâ€”are there any seasonal or culturally meaningful dishes I should try?"
)


await chat_and_show(
    "I'm visiting Kyoto in early spring and I prefer light, vegetable-based dishes. "
    "I also follow Halal dietary rules. Are there any traditional foods or temple-style meals I should try?"
)


await chat_and_show(
    "I just returned from Japan and I really loved strawberry daifuku. "
    "I live in Germany and it's difficult to find Japanese glutinous rice flour. "
    "How can I recreate something similar at home while keeping the traditional texture and feel?"
)


# This example demonstrates:
# - multi-turn conversation using runner.run_debug()
# - ADK session memory preserving user preferences (wheat allergy + mild flavors)
# - cultural and safety-aware filtering for Kyushu regional cuisine

multi_turn_messages = [
    "I'm traveling to Kyushu next month with my 8-year-old child. "
    "They have a wheat allergy and prefer mild flavors. "
    "Could you recommend some traditional foods that would be safe for them?",

    "Thank you! Are any of those dishes commonly served at local festivals or seasonal events in Kyushu?"
]

# Run the agent in debug mode with a list of user messages
events = await runner.run_debug(multi_turn_messages)
final_answer = extract_final_text(events)

# Show all user turns in order
user_lines = []
for i, msg in enumerate(multi_turn_messages, start=1):
    user_lines.append(f"{i}. {msg}")

display(Markdown("### ğŸ§‘ User (multi-turn)\n\n" + "\n\n".join(user_lines)))

# Show the final agent answer based on the full context
display(Markdown("### ğŸ¤– UserGuideAgent (final answer)\n\n" + final_answer))

# Optional: observability
logger.info(
    "Multi-turn example executed with %d user messages.",
    len(multi_turn_messages),
)

