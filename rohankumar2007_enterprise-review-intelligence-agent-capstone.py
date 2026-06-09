import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# =========================
# SECTION 1: Imports & Retry Config
# =========================

import uuid
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

print("âœ… ADK imports loaded.")

# Same retry pattern as in A2A notebook
retry_config = types.HttpRetryOptions(
    attempts=5,       # Maximum retry attempts
    exp_base=7,       # Delay multiplier
    initial_delay=1,  # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Retry config created.")



# =========================
# SECTION 2: Dummy B2B + B2C Feedback Data
# =========================

feedback_data = [
    {
        "id": 1,
        "source": "b2c",
        "channel": "app_review",
        "text": "The app is good but the payment keeps failing at checkout."
    },
    {
        "id": 2,
        "source": "b2b",
        "channel": "client_email",
        "text": "Our team is facing frequent downtime during peak business hours. This is affecting our SLAs."
    },
    {
        "id": 3,
        "source": "b2c",
        "channel": "website_feedback",
        "text": "Customer support took too long to respond to my issue."
    },
    {
        "id": 4,
        "source": "b2b",
        "channel": "support_ticket",
        "text": "We need better analytics visibility and faster resolution for critical incidents."
    },
]

print(f"âœ… Total feedback items loaded: {len(feedback_data)}")



# =========================
# SECTION 3: Tools (Sentiment + Category)
# =========================

def basic_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """
    Very simple rule-based sentiment classifier.
    """
    text_lower = text.lower()
    negative_words = ["bad", "slow", "terrible", "hate", "issue", "downtime", "fail", "failing"]
    positive_words = ["love", "great", "awesome", "excellent", "good", "happy"]

    if any(word in text_lower for word in negative_words):
        return "negative"
    if any(word in text_lower for word in positive_words):
        return "positive"
    return "neutral"


def basic_category(text: str) -> str:
    """
    Coarse issue category based on simple keyword rules.
    """
    text_lower = text.lower()
    if "payment" in text_lower or "checkout" in text_lower:
        return "payment"
    if "support" in text_lower or "respond" in text_lower:
        return "customer_support"
    if "downtime" in text_lower or "sla" in text_lower:
        return "stability_reliability"
    if "analytics" in text_lower or "visibility" in text_lower:
        return "analytics_reporting"
    return "other"


# These are the tools we'll expose to the B2C/B2B agents
def sentiment_tool(text: str) -> str:
    """
    Tool: Analyze sentiment (positive/negative/neutral) for the given feedback.
    """
    return basic_sentiment(text)


def category_tool(text: str) -> str:
    """
    Tool: Classify the feedback text into a coarse issue category.
    """
    return basic_category(text)


print("âœ… Tools ready: sentiment_tool, category_tool")



# =========================
# SECTION 4: Agents (B2C, B2B, Insight)
# =========================

# Shared Gemini model (same pattern as course notebooks)
gemini_model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# 4.1 B2C Agent
b2c_agent = LlmAgent(
    model=gemini_model,
    name="b2c_review_agent",
    description="Analyzes B2C product reviews from end users.",
    instruction=(
        "You analyze B2C customer reviews for a digital product.\n"
        "For the given review text, you MUST:\n"
        "1) Identify the main issue in one sentence.\n"
        "2) Use the sentiment_tool and category_tool to decide sentiment and category.\n"
        "3) Suggest ONE clear product or support improvement.\n\n"
        "Return a short JSON-style string with keys: issue, sentiment, category, suggestion."
    ),
    tools=[sentiment_tool, category_tool],
)

# 4.2 B2B Agent
b2b_agent = LlmAgent(
    model=gemini_model,
    name="b2b_feedback_agent",
    description="Analyzes B2B client feedback from enterprise customers.",
    instruction=(
        "You analyze B2B client feedback such as enterprise emails or tickets.\n"
        "For the given feedback text, provide a JSON-style string with keys:\n"
        "- summary: one-sentence summary of the core problem\n"
        "- impact: low/medium/high business impact\n"
        "- risk: low/medium/high risk to relationship or SLAs\n"
        "- urgency: low/medium/high urgency\n"
        "You may also consider the sentiment_tool as a signal of severity."
    ),
    tools=[sentiment_tool],
)

# 4.3 Insight Agent
insight_agent = LlmAgent(
    model=gemini_model,
    name="insight_aggregator_agent",
    description="Combines B2B and B2C analysis into unified business insights.",
    instruction=(
        "You are an insights analyst combining B2B and B2C feedback analysis.\n"
        "You will receive two Python-like lists: 'b2c_results' and 'b2b_results'.\n"
        "Each item contains raw_text and analysis (JSON-style strings).\n\n"
        "Your task:\n"
        "- Identify top recurring issues from B2C users.\n"
        "- Identify top recurring issues from B2B clients.\n"
        "- Highlight any overlap or pattern between B2B and B2C.\n"
        "- Propose 5 clear action items for product / engineering / support teams.\n\n"
        "Write the final answer as structured bullet points, easy for business stakeholders."
    ),
)

print("âœ… Agents created: b2c_agent, b2b_agent, insight_agent")



# =========================
# SECTION 5: Async Multi-Agent Pipeline
# =========================

async def analyze_feedback_pipeline(feedback_list):
    """
    Multi-agent pipeline using the SAME async pattern as the A2A test function:
    - Uses InMemorySessionService
    - Creates sessions via await create_session(...)
    - Uses Runner.run_async(...) for each agent
    """
    app_name = "review_app"
    user_id = "demo_user"

    # 1) Session management (same as test_a2a_communication)
    session_service = InMemorySessionService()

    # Separate sessions for each logical agent
    session_b2c = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"b2c_session_{uuid.uuid4().hex[:8]}",
    )

    session_b2b = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"b2b_session_{uuid.uuid4().hex[:8]}",
    )

    session_insight = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"insight_session_{uuid.uuid4().hex[:8]}",
    )

    # 2) Runners for each agent
    b2c_runner = Runner(
        agent=b2c_agent,
        app_name=app_name,
        session_service=session_service,
    )

    b2b_runner = Runner(
        agent=b2b_agent,
        app_name=app_name,
        session_service=session_service,
    )

    insight_runner = Runner(
        agent=insight_agent,
        app_name=app_name,
        session_service=session_service,
    )

    # Helper: run an agent and get the final text response (same pattern as A2A notebook)
    async def run_llm(runner: Runner, session_id: str, text: str) -> str:
        content = types.Content(parts=[types.Part(text=text)])
        final_text = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text"):
                        final_text = part.text

        return final_text

    # 3) Run B2C + B2B analysis
    b2c_results = []
    b2b_results = []

    for fb in feedback_list:
        text = fb["text"]

        if fb["source"] == "b2c":
            analysis = await run_llm(
                b2c_runner,
                session_id=session_b2c.id,
                text=text,
            )
            b2c_results.append(
                {
                    "id": fb["id"],
                    "channel": fb["channel"],
                    "raw_text": text,
                    "analysis": analysis,
                }
            )

        elif fb["source"] == "b2b":
            analysis = await run_llm(
                b2b_runner,
                session_id=session_b2b.id,
                text=text,
            )
            b2b_results.append(
                {
                    "id": fb["id"],
                    "channel": fb["channel"],
                    "raw_text": text,
                    "analysis": analysis,
                }
            )

    # 4) Combine into a prompt for the insight agent
    combined_prompt = (
        "You are the insights agent.\n\n"
        "Here are B2C analysis results:\n"
        f"{b2c_results}\n\n"
        "Here are B2B analysis results:\n"
        f"{b2b_results}\n\n"
        "Now produce a unified insight report with:\n"
        "- Top B2C issues\n"
        "- Top B2B issues\n"
        "- Any overlaps\n"
        "- 5 clear action items.\n"
    )

    insight_report = await run_llm(
        insight_runner,
        session_id=session_insight.id,
        text=combined_prompt,
    )

    return {
        "b2c_results": b2c_results,
        "b2b_results": b2b_results,
        "insight_report": insight_report,
    }

print("âœ… Async pipeline analyze_feedback_pipeline() defined.")



# =========================
# SECTION 6: Run the Pipeline
# =========================

# IMPORTANT: Use await (like you did in A2A notebook)

result = await analyze_feedback_pipeline(feedback_data)

print("\n===== SAMPLE B2C ANALYSIS =====")
for item in result["b2c_results"]:
    print(f"\nID: {item['id']} | Channel: {item['channel']}")
    print("Raw:", item["raw_text"])
    print("Analysis:", item["analysis"])

print("\n\n===== SAMPLE B2B ANALYSIS =====")
for item in result["b2b_results"]:
    print(f"\nID: {item['id']} | Channel: {item['channel']}")
    print("Raw:", item["raw_text"])
    print("Analysis:", item["analysis"])

print("\n\n===== UNIFIED INSIGHT REPORT =====")
print(result["insight_report"])


