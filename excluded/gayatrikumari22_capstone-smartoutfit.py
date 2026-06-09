import os
from kaggle_secrets import UserSecretsClient

# Load Gemini API Key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print("â�Œ ERROR: Add GOOGLE_API_KEY to Kaggle Secrets.", e)


# ADK imports
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
print("âœ… ADK components imported successfully.")


def manual_outfit_lookup(occasion: str) -> dict:
    """
    Hybrid tool â€“ tries to find outfit in dataset.
    If not found, Gemini LLM will take over automatically.
    """

    outfit_dataset = {
        "casual": "Oversized T-shirt, high-waisted jeans, white sneakers.",
        "office": "Formal shirt, tailored trousers, loafers, minimalist watch.",
        "beach": "Floral crop top, flowy shorts, flip-flops, sunglasses.",
        "wedding": "Lehenga, designer blouse, kundan jewellery, golden heels.",
    }

    result = outfit_dataset.get(occasion.lower())

    if result:
        return {"status": "success", "outfit": result}

    return {"status": "error", "error_message": "not found"}


print("âœ… Manual outfit tool created.")


# ======================================================
#  HYBRID AGENT (Dataset First â†’ Gemini Fallback)
# ======================================================

root_agent = Agent(
    name="hybrid_outfit_assistant",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(
            attempts=5,
            exp_base=7,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )
    ),
    description="A hybrid stylist: dataset first, LLM fallback.",
    instruction=(
        "You are a top-class stylist.\n"
        "1ï¸�âƒ£ FIRST: Always call the tool 'manual_outfit_lookup'.\n"
        "2ï¸�âƒ£ If tool returns error, THEN use reasoning and generate creative outfits.\n"
        "3ï¸�âƒ£ Give detailed, stylish, practical suggestions.\n"
        "4ï¸�âƒ£ If user asks multiple occasions, answer all.\n"
    ),
    tools=[manual_outfit_lookup],
)

print("âœ… Hybrid agent created.")



# ======================================================
# 4. RUNNER + QUERY (Kaggle-compatible)
# ======================================================

async def ask_ai(query: str):
    runner = InMemoryRunner(agent=root_agent)
    return await runner.run_debug(query)


# Example: query that dataset does NOT have
response = await ask_ai("Suggest stylish outfits for nightclub and beach together.")

print("\nğŸ¤– FINAL RESPONSE:\n", response)


