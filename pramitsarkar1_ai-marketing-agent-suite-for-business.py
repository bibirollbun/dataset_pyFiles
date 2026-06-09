# ============================
# 2ï¸�âƒ£ Google ADK Setup (Imports + Retry)
# ============================

from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext, preload_memory
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.memory import InMemoryMemoryService

import os
import json
import textwrap

import io
import sys
import contextlib


# Retry configuration (same style as course notebooks)
retry_config = types.HttpRetryOptions(
    attempts=5,            # Retry up to 5 times
    exp_base=7,            # Backoff multiplier
    initial_delay=1,       # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

print("âœ… Google ADK imports & retry config ready")


# ============================
# 3ï¸�âƒ£ Root Agent + Runner Setup (ADK)
# ============================

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

APP_NAME = "smm_suite_app"
USER_ID = "demo_user"
SESSION_ID = "default_session"

MODEL_NAME = "gemini-2.5-flash-lite"

# 1ï¸�âƒ£ Create the root LLM agent (simple, without memory yet â€“ we rebuild later with memory)
root_agent = LlmAgent(
    name="smm_root_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are the coordinator of an AI Social Media Marketing Agent Suite for agencies. "
        "You will later delegate work to specialist agents including: "
        "business Strategy, Content Writing, Hashtags & SEO, Design Prompts, Calendar Planning, "
        "Analytics Insights, and Review & QA."
    ),
    tools=[],  # tools added later / we rebuild with memory later
)

print("âœ… Root LLM agent created (smm_root_agent)")

# 2ï¸�âƒ£ Set up Session Memory
session_service = InMemorySessionService()
print("âœ… InMemorySessionService initialized")

# 3ï¸�âƒ£ Create the Runner
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

print("âœ… Runner initialized with:")
print(f"   â€¢ APP_NAME   = {APP_NAME}")
print(f"   â€¢ USER_ID    = {USER_ID}")
print(f"   â€¢ SESSION_ID = {SESSION_ID}")


# ============================
# ğŸ”‘  Google API Key Setup (Required)
# ============================

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key

print("âœ… Google API Key loaded into environment")


###################################
@contextlib.contextmanager
def capture_print():
    """
    Temporarily capture all print() output into a buffer.
    Use:
      with capture_print() as buf:
          print("hello")
      text = buf.getvalue()
    """
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        yield buf
    finally:
        sys.stdout = old_stdout

##################################

# ============================
# business Strategy Agent
# ============================

business_strategy_agent = LlmAgent(
    name="business_strategy_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a senior Social Media Marketing Strategist for agencies. "
        "Given a client brief, you MUST return a clear business strategy in structured JSON.\n\n"
        "Your JSON MUST use this structure:\n"
        "{\n"
        '  \"business_summary\": \"short summary of the business\",\n'
        '  \"business_voice\": \"description of the tone and style\",\n'
        '  \"do_donts\": [\"Do: ...\", \"Don\'t: ...\"],\n'
        '  \"content_pillars\": [\n'
        '    {\"name\": \"Education\", \"description\": \"...\"},\n'
        '    {\"name\": \"Engagement\", \"description\": \"...\"},\n'
        '    {\"name\": \"Promotion\", \"description\": \"...\"}\n'
        '  ],\n'
        '  \"posting_guidelines\": [\"Guideline 1\", \"Guideline 2\"],\n'
        '  \"primary_kpis\": [\"Reach\", \"Saves\", \"Clicks\"]\n'
        "}\n\n"
        "Always answer in valid JSON only. No extra text."
    ),
    tools=[],
)

print("âœ… Business Strategy Agent created: business_strategy_agent")


# ============================
# 6ï¸�âƒ£ Content Writer Agent
# ============================

content_writer_agent = LlmAgent(
    name="content_writer_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a professional social media content writer working for an agency.\n"
        "You will receive:\n"
        "  - A business strategy JSON object (with business_voice, content_pillars, posting_guidelines, etc.)\n"
        "  - A number of days to plan content for\n"
        "  - A list of platforms (e.g., Instagram, Facebook)\n\n"
        "You MUST return a JSON array of posts.\n"
        "Each post MUST have this structure:\n"
        "[\n"
        "  {\n"
        "    \"day\": 1,\n"
        "    \"pillar\": \"Education\",\n"
        "    \"idea\": \"Short description of the concept of the post\",\n"
        "    \"platform_copies\": {\n"
        "      \"Instagram\": \"Caption tailored for Instagram...\",\n"
        "      \"Facebook\": \"Caption tailored for Facebook...\"\n"
        "    }\n"
        "  },\n"
        "  { ... }\n"
        "]\n\n"
        "Rules:\n"
        "- Always follow the business_voice.\n"
        "- Use the content_pillars names in the 'pillar' field.\n"
        "- Vary hooks and CTAs.\n"
        "- Do NOT include hashtags here (another agent will do that).\n"
        "- Always output VALID JSON only. No extra commentary."
    ),
    tools=[],
)

print("âœ… Content Writer Agent created: content_writer_agent")


# ============================
# 7ï¸�âƒ£ Hashtag & SEO Agent
# ============================

hashtag_agent = LlmAgent(
    name="hashtag_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a social media SEO & hashtag expert for marketing agencies.\n"
        "You will receive:\n"
        "  - A JSON array of posts (with day, pillar, idea, platform_copies).\n\n"
        "Your task:\n"
        "- For EACH post, generate 10â€“20 relevant hashtags.\n"
        "- Mix niche + broad hashtags.\n"
        "- Align with the business and pillar.\n"
        "- Do NOT repeat the same exact set for all posts.\n\n"
        "You MUST return a JSON array where each item has:\n"
        "[\n"
        "  {\n"
        "    \"day\": 1,\n"
        "    \"hashtags\": [\"#example1\", \"#example2\", \"#...\"]\n"
        "  },\n"
        "  { ... }\n"
        "]\n\n"
        "Rules:\n"
        "- Do NOT include caption text.\n"
        "- Only output VALID JSON, no extra explanation."
    ),
    tools=[],
)

print("âœ… Hashtag Agent created: hashtag_agent")


# ============================
# 8ï¸�âƒ£ Design Prompt Agent
# ============================

design_prompt_agent = LlmAgent(
    name="design_prompt_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a creative director for social media design.\n"
        "You will receive a JSON array of posts (day, pillar, idea, platform_copies).\n\n"
        "For EACH post, you MUST generate a short but clear design brief.\n"
        "Return a JSON array where each item has:\n"
        "[\n"
        "  {\n"
        "    \"day\": 1,\n"
        "    \"design_brief\": \"One paragraph description of the visual concept, layout, style, and on-image text.\"\n"
        "  },\n"
        "  { ... }\n"
        "]\n\n"
        "Rules:\n"
        "- Tailor the design to the pillar and idea.\n"
        "- Mention if it's a reel/short, carousel, or single image.\n"
        "- Mention key on-image headline text if relevant.\n"
        "- Output VALID JSON only. No extra explanation."
    ),
    tools=[],
)

print("âœ… Design Prompt Agent created: design_prompt_agent")


# ============================
# 9ï¸�âƒ£ Calendar Agent
# ============================

calendar_agent = LlmAgent(
    name="calendar_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a social media content planner for marketing agencies.\n"
        "You will receive a JSON array of posts generated by another agent.\n"
        "Each post has fields like: day, pillar, idea, platform_copies.\n\n"
        "Your task:\n"
        "- Organize these posts into a clear calendar JSON structure.\n"
        "- If multiple posts share the same 'day', group them under that day.\n"
        "- You MUST NOT change the text of posts.\n\n"
        "Output format:\n"
        "{\n"
        "  \"days\": [\n"
        "    {\n"
        "      \"day\": 1,\n"
        "      \"posts\": [\n"
        "        {\n"
        "          \"pillar\": \"...\",\n"
        "          \"idea\": \"...\",\n"
        "          \"platform_copies\": {\"Instagram\": \"...\", \"Facebook\": \"...\"}\n"
        "        }\n"
        "      ]\n"
        "    },\n"
        "    {\"day\": 2, \"posts\": [ ... ]}\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Do NOT invent new posts.\n"
        "- Do NOT add hashtags or design briefs here (other agents handle that).\n"
        "- Always output VALID JSON only, no extra explanation."
    ),
    tools=[],
)

print("âœ… Calendar Agent created: calendar_agent")


# ============================
# ğŸ”Ÿ Analytics Insights Agent
# ============================

analytics_agent = LlmAgent(
    name="analytics_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a social media analytics strategist for marketing agencies.\n"
        "You will receive:\n"
        "  - A business strategy JSON object (business_summary, business_voice, content_pillars, posting_guidelines, primary_kpis, etc.)\n"
        "  - A calendar JSON object (days -> posts with pillar, idea, platform_copies)\n\n"
        "Your job is to analyze the PLAN (not real performance data) and return structured INSIGHTS.\n\n"
        "You MUST return a single JSON object with the following keys:\n"
        "{\n"
        "  \"summary\": \"High-level summary of the calendar and alignment with goals.\",\n"
        "  \"strengths\": [\"Point 1\", \"Point 2\", \"...\"],\n"
        "  \"risks\": [\"Point 1\", \"Point 2\", \"...\"],\n"
        "  \"recommendations\": [\"Actionable suggestion 1\", \"...\"],\n"
        "  \"suggested_metrics\": [\"Metric 1\", \"Metric 2\", \"...\"]\n"
        "}\n\n"
        "Guidelines:\n"
        "- Be specific, not generic.\n"
        "- Use the content_pillars and posting_guidelines to judge balance and variety.\n"
        "- Use primary_kpis to propose relevant suggested_metrics.\n"
        "- Do NOT include any text outside of the JSON object.\n"
        "- Always output STRICTLY VALID JSON only."
    ),
    tools=[],
)

print("âœ… Analytics Insights Agent created: analytics_agent")


# ============================
# âœ… Review & QA Agent
# ============================

review_qa_agent = LlmAgent(
    name="review_qa_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are a senior Social Media Director doing final quality review before sending a plan to a client.\n"
        "You will receive:\n"
        "  - business strategy JSON\n"
        "  - Calendar JSON\n"
        "  - Analytics insights JSON\n\n"
        "You MUST return a JSON object with this structure:\n"
        "{\n"
        "  \"overall_score\": 0-100,\n"
        "  \"issues\": [\"Issue 1\", \"Issue 2\", \"...\"],\n"
        "  \"suggestions\": [\"Suggestion 1\", \"Suggestion 2\", \"...\"],\n"
        "  \"risk_level\": \"low\" | \"medium\" | \"high\",\n"
        "  \"is_client_ready\": true or false\n"
        "}\n\n"
        "Guidelines:\n"
        "- Look for missing pillars, too many promotions, weak CTAs, or unclear audience targeting.\n"
        "- Be tough but fair.\n"
        "- Always output VALID JSON and nothing else."
    ),
    tools=[],
)

print("âœ… Review & QA Agent created: review_qa_agent")


# ============================
# ğŸ“§ Email Campaign Agent (Daily Emails)
# ============================

email_agent = LlmAgent(
    name="email_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are an email marketing specialist working for a social media marketing agency.\n"
        "You write email campaigns that the CLIENT will send to THEIR CUSTOMERS (the client's clients).\n\n"
        "You will often be asked to create emails for a specific number of days (e.g. 7 days, 30 days).\n"
        "When the user says 'for the next N days', you MUST create EXACTLY N emails:\n"
        "- one email for each day from 1 to N\n"
        "- no missing days\n"
        "- no extra days\n\n"
        "You MUST ALWAYS respond with ONLY VALID JSON (no markdown, no backticks) in this exact shape:\n"
        "[\n"
        "  {\n"
        "    \"day\": 1,\n"
        "    \"email_type\": \"newsletter | promotion | reminder | announcement | onboarding\",\n"
        "    \"subject\": \"Short, high-converting subject line\",\n"
        "    \"preview_text\": \"Short inbox preview line (40â€“80 chars)\",\n"
        "    \"audience_segment\": \"Who this email is for (e.g. all customers, new customers, regulars, students)\",\n"
        "    \"body_text\": \"Plain-text email body (no HTML). 120â€“300 words, paragraphs separated by blank lines.\",\n"
        "    \"primary_cta\": \"Clear main call to action (e.g. Visit the cafe this weekend, Book a trial, Shop now)\"\n"
        "  }\n"
        "]\n\n"
        "Guidelines:\n"
        "- Align with the client's tone of voice and goals.\n"
        "- Use friendly, clear language that matches the business.\n"
        "- Vary email_type across the period (not all promotions).\n"
        "- Make CTAs specific and action-oriented.\n"
        "- Never include any explanation outside the JSON array.\n"
    ),
    tools=[],
)

print('âœ… Email Campaign Agent created: email_agent (daily emails, JSON-only)')


# ============================
# ğŸŒ� Trend Research Agent (Google Search + Strict JSON)
# ============================

trend_research_agent = LlmAgent(
    name="trend_research_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[google_search],  # âœ… uses Google Search tool
    instruction=(
        "You are the Trend Research Agent for a social media marketing agency.\n"
        "You MUST use the google_search tool when appropriate to gather up-to-date trends.\n\n"
        "You ALWAYS respond with ONE SINGLE VALID JSON OBJECT and NOTHING ELSE.\n"
        "Important rules:\n"
        "- Your reply MUST start with '{' and end with '}'.\n"
        "- All KEYS must be in double quotes, e.g. \"trend_summary\".\n"
        "- All STRING values must be in double quotes.\n"
        "- NEVER write bare lines like: trend_summary: \"...\" (that is INVALID).\n"
        "- NEVER wrap your JSON in ``` or ```json fences.\n"
        "- If you include quotes inside text, escape them or rephrase to avoid extra quotes.\n\n"
        "The JSON object MUST have EXACTLY these keys:\n"
        "{\n"
        "  \"trend_summary\": \"1â€“3 paragraph overview of relevant social + marketing trends\",\n"
        "  \"trend_topics\": [\"short bullet topics like TikTok Food Trends\", \"...\"],\n"
        "  \"content_ideas\": [\"list of practical content ideas for this business\", \"...\"],\n"
        "  \"suggested_keywords\": [\"seo-style keyword or phrase\", \"...\"]\n"
        "}\n\n"
        "If you respond in ANY other format, the tool will crash, so you MUST obey the JSON rules above."
    ),
)

print('âœ… Trend Research Agent created: trend_research_agent (Google Search + strict JSON)')



# ============================
# ğŸ§  Agent Memory for SMM (Sessions + Long-Term)
# ============================

print("âœ… ADK memory components imported successfully.")

# Long-term memory service
memory_service = InMemoryMemoryService()

# Session service (we already created above, but ensure exists)
try:
    session_service
except NameError:
    session_service = InMemorySessionService()

try:
    APP_NAME
except NameError:
    APP_NAME = "smm_suite_app"

try:
    USER_ID
except NameError:
    USER_ID = "demo_user"


# ============================
# ğŸ§  Auto-save to Memory Callback + SMM Memory Agent
# ============================

async def auto_save_to_memory(callback_context):
    """Automatically save the last session turn into long-term memory."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("âœ… auto_save_to_memory callback created.")

smm_memory_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="SmmMemoryAgent",
    instruction=(
        "You are an SMM assistant that remembers client details over time.\n"
        "Use memory to recall client name, business, tone, goals, and preferences.\n"
        "Answer in simple, helpful language."
    ),
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory,
)

print("âœ… SmmMemoryAgent created with automatic memory saving.")

smm_memory_runner = Runner(
    agent=smm_memory_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… Runner created for SmmMemoryAgent (session + memory enabled).")


# ============================
# ğŸ”� Helper â€“ Run a Session with Memory
# ============================

async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Run one or more user queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except Exception:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(
            role="user",
            parts=[types.Part(text=query)],
        )

        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=query_content,
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text != "None":
                        print(f"Model > {text}")

print("âœ… run_session helper defined.")


# ============================
# ğŸ§  Root Coordinator Agent with Memory
# ============================

print("âœ… Imports for root agent with memory are ready.")

try:
    session_service
except NameError:
    session_service = InMemorySessionService()

try:
    memory_service
except NameError:
    memory_service = InMemoryMemoryService()

try:
    APP_NAME
except NameError:
    APP_NAME = "smm_suite_app"

try:
    USER_ID
except NameError:
    USER_ID = "demo_user"

try:
    MODEL_NAME
except NameError:
    MODEL_NAME = "gemini-2.5-flash-lite"


async def auto_save_to_memory_root(callback_context):
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("âœ… auto_save_to_memory callback (for root agent) is defined.")

root_agent = LlmAgent(
    name="smm_root_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction=(
        "You are the ROOT coordinator of the AI Social Media Marketing Agent Suite for agencies.\n"
        "You understand client briefs, remember client details over time, and orchestrate specialist agents:\n"
        "- Trend Research (Google Search)\n"
        "- Business Strategy\n"
        "- Content Writing\n"
        "- Hashtags & SEO\n"
        "- Design Prompts\n"
        "- Calendar Planning\n"
        "- Email Campaigns\n"
        "- Analytics Insights\n"
        "- Review & QA\n\n"
        "Use memory to recall client name, tone of voice, goals, and important preferences across turns."
    ),
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory_root,
)

print("âœ… Root LLM agent rebuilt with memory: smm_root_agent")

root_runner_with_memory = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… root_runner_with_memory created (sessions + memory enabled).")


# ============================
# ğŸ§¾ Human-Readable Full Plan Printer (now with Trend Insights)
# ============================

def print_full_smm_plan_readable(
    business_strategy: dict,
    content_posts: list,
    hashtag_list: list,
    design_list: list,
    calendar_obj: dict,
    analytics_obj: dict,
    review_qa_obj: dict,
    email_list: list | None = None,
    trend_insights: dict | None = None,
):
    # 1ï¸�âƒ£ Build helper maps: day -> hashtags / design
    hashtags_by_day = {item["day"]: item["hashtags"] for item in hashtag_list}
    design_by_day = {item["day"]: item["design_brief"] for item in design_list}

    posts_enriched = []
    for post in content_posts:
        day = post.get("day")
        post["hashtags"] = hashtags_by_day.get(day, [])
        post["design_brief"] = design_by_day.get(day, "")
        posts_enriched.append(post)

    # ========== 0. TREND INSIGHTS (Google Search) ==========
    if trend_insights is not None:
        print("\n" + "="*90)
        print("ğŸŒ� 0) TREND INSIGHTS (from Google Search)")
        print("="*90)

        print("\nTrend Summary:")
        print(" ", trend_insights.get("trend_summary", "â€”"))

        print("\nTrend Topics:")
        for t in trend_insights.get("trend_topics", []):
            print(" -", t)

        print("\nContent Ideas from Trends:")
        for idea in trend_insights.get("content_ideas", []):
            print(" -", idea)

        print("\nSuggested Keywords:")
        for kw in trend_insights.get("suggested_keywords", []):
            print(" -", kw)

    # ========== 1. Business STRATEGY ==========
    print("\n" + "="*90)
    print("ğŸ§  1) Business STRATEGY")
    print("="*90)

    print("\nSummary:")
    print(" ", business_strategy.get("business_summary", "â€”"))

    print("\nVoice:")
    print(" ", business_strategy.get("business_voice", "â€”"))

    print("\nDo & Don'ts:")
    for rule in business_strategy.get("do_donts", []):
        print(" -", rule)

    print("\nContent Pillars:")
    for pillar in business_strategy.get("content_pillars", []):
        print(f" â€¢ {pillar.get('name', 'Unnamed Pillar')}:")
        print("    ", pillar.get("description", ""))

    print("\nPosting Guidelines:")
    for g in business_strategy.get("posting_guidelines", []):
        print(" -", g)

    print("\nPrimary KPIs:")
    for k in business_strategy.get("primary_kpis", []):
        print(" -", k)

    # ========== 2. CONTENT + DESIGN + HASHTAGS ==========
    print("\n" + "="*90)
    print("âœ�ï¸� 2) CONTENT POSTS + ğŸ�¨ DESIGN PROMPTS + #ï¸�âƒ£ HASHTAGS")
    print("="*90)

    for post in posts_enriched:
        day = post.get("day")
        pillar = post.get("pillar")
        idea = post.get("idea")
        copies = post.get("platform_copies", {})
        design_brief = post.get("design_brief", "")
        hashtags = post.get("hashtags", [])

        print(f"\n--- Day {day} â€” Pillar: {pillar} ---")
        print("Idea:")
        print(" ", idea)

        ig = copies.get("Instagram")
        fb = copies.get("Facebook")
        if ig:
            print("\n  ğŸŸ£ Instagram Caption:")
            print("   ", ig)
        if fb:
            print("\n  ğŸ”µ Facebook Caption:")
            print("   ", fb)

        print("\n  ğŸ�¨ Design Prompt:")
        print("   ", design_brief)

        print("\n  #ï¸�âƒ£ Hashtags:")
        print("   ", " ".join(hashtags))

    # ========== 3. CALENDAR OVERVIEW ==========
    print("\n" + "="*90)
    print("ğŸ“… 3) CONTENT CALENDAR OVERVIEW")
    print("="*90)

    for day_block in calendar_obj.get("days", []):
        d = day_block.get("day")
        posts = day_block.get("posts", [])
        print(f"\nDay {d}: {len(posts)} post(s)")
        for p in posts:
            print(" - Pillar:", p.get("pillar", "â€”"), "| Idea:", p.get("idea", "â€”"))

    # ========== 4. EMAIL CAMPAIGN (DAILY) ==========
    if email_list is not None:
        print("\n" + "="*90)
        print("ğŸ“§ 4) DAILY EMAIL CAMPAIGN (CLIENT â†’ THEIR CUSTOMERS)")
        print("="*90)

        for email in email_list:
            print(f"\n--- Day {email.get('day', 'â€”')} â€” {email.get('email_type', '').upper()} ---")
            print("Subject      :", email.get("subject", "â€”"))
            print("Preview Text :", email.get("preview_text", "â€”"))
            print("Audience     :", email.get("audience_segment", "â€”"))
            print("\nBody:\n", email.get("body_text", "â€”"))
            print("\nPrimary CTA  :", email.get("primary_cta", "â€”"))
            print("-" * 60)

    # ========== 5. ANALYTICS ==========
    print("\n" + "="*90)
    print("ğŸ“Š 5) ANALYTICS INSIGHTS")
    print("="*90)

    print("\nSummary:")
    print(" ", analytics_obj.get("summary", "â€”"))

    print("\nStrengths:")
    for s in analytics_obj.get("strengths", []):
        print(" -", s)

    print("\nRisks:")
    for r in analytics_obj.get("risks", []):
        print(" -", r)

    print("\nRecommendations:")
    for rec in analytics_obj.get("recommendations", []):
        print(" -", rec)

    print("\nSuggested Metrics:")
    for m in analytics_obj.get("suggested_metrics", []):
        print(" -", m)

    # ========== 6. REVIEW & QA ==========
    print("\n" + "="*90)
    print("âœ… 6) REVIEW & QA SUMMARY")
    print("="*90)

    print("\nOverall score:", review_qa_obj.get("overall_score", "N/A"))
    print("Risk level:", review_qa_obj.get("risk_level", "N/A"))
    print("Is client-ready?:", review_qa_obj.get("is_client_ready", "N/A"))

    print("\nIssues:")
    for issue in review_qa_obj.get("issues", []):
        print(" -", issue)

    print("\nSuggestions:")
    for sug in review_qa_obj.get("suggestions", []):
        print(" -", sug)


# ============================
# ğŸ”§ Helper: call agent via run_debug() and parse JSON
# ============================

async def call_agent_and_get_json(runner, prompt: str, label: str = "agent"):
    events = await runner.run_debug(prompt)

    final_text = None
    for ev in reversed(events):
        content = getattr(ev, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts and getattr(parts[0], "text", None):
            final_text = parts[0].text
            break

    if not final_text:
        raise ValueError(f"â�Œ Could not extract text output for {label}.")

    clean = final_text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines[0].strip().lower().startswith("```json"):
            lines = lines[1:]
        elif lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    try:
        return json.loads(clean)
    except Exception as e:
        print(f"\nâ�Œ Failed to parse JSON for {label}: {e}")
        print("Raw text was:\n", clean)
        raise


# ============================
# ğŸš€ Orchestrator: run_smm_suite_for_client (now calls Trend Agent)
# ============================

async def run_smm_suite_for_client(client_brief_text: str, days: int = 7):
    """
    End-to-end run with HUMAN-READABLE summary at the end.

    0) Root coordinator (with memory) reads & stores the client brief
    1) Trend Research Agent (Google Search)
    2) Business Strategy Agent
    3) Content Writer Agent
    4) Hashtag & SEO Agent
    5) Design Prompt Agent
    6) Calendar Agent
    7) Email Campaign Agent (client -> their customers)
    8) Analytics Agent
    9) Review & QA Agent
    """

    print("\n" + "="*100)
    print("ğŸ§  STEP 0 â€” ROOT COORDINATOR READS & REMEMBERS CLIENT BRIEF")
    print("="*100)

    global USER_ID, APP_NAME, session_service
    if "USER_ID" not in globals():
        USER_ID = "demo_user"
    if "APP_NAME" not in globals():
        APP_NAME = "smm_suite_app"
    if "session_service" not in globals():
        session_service = InMemorySessionService()

    session_id = "smm-root-memory-session"
    try:
        root_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
    except Exception:
        root_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )

    root_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=textwrap.dedent(f"""
                You are the ROOT coordinator of the SMM Agent Suite.

                Here is a NEW client brief. Read it carefully and store the key details
                (client name, industry, tone, goals, platforms) in memory for future tasks.

                Client brief:
                {client_brief_text}

                Reply with a short confirmation summarizing what you understood and that you saved it to memory.
                """)
            )
        ],
    )

    async for event in root_runner_with_memory.run_async(
        user_id=USER_ID,
        session_id=root_session.id,
        new_message=root_message,
    ):
        if hasattr(event, "is_final_response") and event.is_final_response():
            if event.content and event.content.parts:
                text_resp = event.content.parts[0].text
                if text_resp and text_resp != "None":
                    print("\nsmm_root_agent >\n", text_resp)

        # 1ï¸�âƒ£ Trend Research Agent (Google Search)
    print("\n" + "="*100)
    print("ğŸŒ� STEP 1 â€” TREND RESEARCH AGENT (GOOGLE SEARCH)")
    print("="*100)

    trend_runner = InMemoryRunner(agent=trend_research_agent)

    trend_prompt = textwrap.dedent(f"""
    You are the Trend Research Agent.

    Use the Google Search tool to research real-time social media and marketing trends
    that are relevant for this client.

    Client brief:
    {client_brief_text}

    You MUST respond with ONLY ONE valid JSON object and NOTHING ELSE.
    Your JSON MUST have EXACTLY these keys:
    - trend_summary
    - trend_topics
    - content_ideas
    - suggested_keywords

    Rules:
    - Response MUST start with '{{' and end with '}}'
    - All keys MUST be in double quotes
    - All string values MUST be in double quotes
    - No markdown, no backticks, no plain key: value lines
    """)

    trend_insights = await call_agent_and_get_json(
        trend_runner,
        trend_prompt,
        "Trend Research"
    )


    # 2ï¸�âƒ£ business Strategy Agent
    print("\n" + "="*100)
    print("ğŸ�¯ STEP 2 â€” Business STRATEGY AGENT")
    print("="*100)

    business_runner = InMemoryRunner(agent=business_strategy_agent)

    business_prompt = textwrap.dedent(f"""
    You are the business Strategy Agent for an SMM agency.

    Read this client brief and generate the FULL business strategy JSON
    (business_summary, business_voice, do_donts, content_pillars, posting_guidelines, primary_kpis).

    Client brief:
    {client_brief_text}

    Return ONLY a JSON object.
    """)

    business_strategy = await call_agent_and_get_json(business_runner, business_prompt, "business Strategy")

    # 3ï¸�âƒ£ Content Writer Agent
    print("\n" + "="*100)
    print("âœ�ï¸� STEP 3 â€” CONTENT WRITER AGENT")
    print("="*100)

    content_runner = InMemoryRunner(agent=content_writer_agent)

    content_prompt = textwrap.dedent(f"""
    You are the Content Writer Agent.

    Based on this client brief, create a {days}-day content plan.
    Platforms: Instagram and Facebook.

    For EACH DAY, create ONE main post:
    - day
    - pillar
    - idea
    - platform_copies (Instagram + Facebook)

    Client brief:
    {client_brief_text}

    Return ONLY a JSON array of posts.
    """)

    content_posts = await call_agent_and_get_json(content_runner, content_prompt, "Content Posts")

    # 4ï¸�âƒ£ Hashtag & SEO Agent
    print("\n" + "="*100)
    print("ğŸ�· STEP 4 â€” HASHTAG & SEO AGENT")
    print("="*100)

    hashtag_runner = InMemoryRunner(agent=hashtag_agent)

    hashtag_prompt = textwrap.dedent(f"""
    You are the Hashtag & SEO Agent.

    For each day from 1 to {days}, create a set of 10â€“20 hashtags.

    Return ONLY a JSON array like:
    [
      {{ "day": 1, "hashtags": ["#...", "#..."] }},
      ...
    ]

    Client brief:
    {client_brief_text}
    """)

    hashtag_list = await call_agent_and_get_json(hashtag_runner, hashtag_prompt, "Hashtags")

    # 5ï¸�âƒ£ Design Prompt Agent
    print("\n" + "="*100)
    print("ğŸ�¨ STEP 5 â€” DESIGN PROMPT AGENT")
    print("="*100)

    design_runner = InMemoryRunner(agent=design_prompt_agent)

    design_prompt = textwrap.dedent(f"""
    You are the Design Prompt Agent.

    Based on the same {days}-day plan, create ONE visual concept per day.

    For EACH DAY, return:
    - day
    - design_brief: a short description of the creative visual

    Client brief:
    {client_brief_text}

    Return ONLY a JSON array.
    """)

    design_list = await call_agent_and_get_json(design_runner, design_prompt, "Design Prompts")

    # 6ï¸�âƒ£ Calendar Agent
    print("\n" + "="*100)
    print("ğŸ—“ STEP 6 â€” CALENDAR AGENT")
    print("="*100)

    calendar_runner = InMemoryRunner(agent=calendar_agent)

    calendar_prompt = textwrap.dedent(f"""
    You are the Calendar Agent.

    Create a structured {days}-day calendar with:
    - days: list of day blocks
    - each day has posts with pillar, idea, platform_copies

    Infer reasonable structure from the brief.

    Client brief:
    {client_brief_text}

    Return ONLY a JSON object with "days": [...]
    """)

    calendar_obj = await call_agent_and_get_json(calendar_runner, calendar_prompt, "Calendar")

    # 7ï¸�âƒ£ Email Campaign Agent
    print("\n" + "="*100)
    print("ğŸ“§ STEP 7 â€” EMAIL CAMPAIGN AGENT (CLIENT â†’ THEIR CUSTOMERS)")
    print("="*100)

    email_runner = InMemoryRunner(agent=email_agent)

    email_prompt = textwrap.dedent(f"""
    You are the Email Marketing Agent.

    Generate an email campaign for EVERY single day from day 1 to day {days}.

    You MUST create EXACTLY {days} emails:
    - one object per day, from 1 to {days}
    - fields: day, email_type, subject, preview_text, audience_segment, body_text, primary_cta

    Client brief:
    {client_brief_text}

    Return ONLY a JSON array.
    """)

    email_list = await call_agent_and_get_json(email_runner, email_prompt, "Emails")

    # 8ï¸�âƒ£ Analytics Agent
    print("\n" + "="*100)
    print("ğŸ“Š STEP 8 â€” ANALYTICS INSIGHTS AGENT")
    print("="*100)

    analytics_runner = InMemoryRunner(agent=analytics_agent)

    analytics_prompt = textwrap.dedent(f"""
    You are the Analytics Insights Agent.

    Based on this client brief and a reasonable {days}-day plan,
    analyze the plan quality and return:

    - summary
    - strengths
    - risks
    - recommendations
    - suggested_metrics

    Client brief:
    {client_brief_text}

    Return ONLY one JSON object.
    """)

    analytics_obj = await call_agent_and_get_json(analytics_runner, analytics_prompt, "Analytics")

    # 9ï¸�âƒ£ Review & QA Agent
    print("\n" + "="*100)
    print("âœ… STEP 9 â€” REVIEW & QA AGENT")
    print("="*100)

    review_runner = InMemoryRunner(agent=review_qa_agent)

    review_prompt = textwrap.dedent(f"""
    You are the Review & QA Agent (senior SMM director).

    Assume:
    - A full {days}-day content plan was created for this client.
    - Calendar + analytics insights exist from other agents.

    Based on the client brief and your expectations for a strong plan,
    return a JSON object with:

    - overall_score (0â€“100)
    - issues (list of strings)
    - suggestions (list of strings)
    - risk_level ("low" | "medium" | "high")
    - is_client_ready (true/false)

    Client brief:
    {client_brief_text}

    Return ONLY the JSON object.
    """)

    review_qa_obj = await call_agent_and_get_json(review_runner, review_prompt, "Review & QA")

    # ğŸ”š Final combined human-readable view
    print("\n\n" + "="*100)
    print("ğŸ§¾ FULL SMM PLAN â€“ HUMAN READABLE VIEW")
    print("="*100)

    # Capture the human-readable output into a buffer
    with capture_print() as buf:
        print_full_smm_plan_readable(
            business_strategy=business_strategy,
            content_posts=content_posts,
            hashtag_list=hashtag_list,
            design_list=design_list,
            calendar_obj=calendar_obj,
            analytics_obj=analytics_obj,
            review_qa_obj=review_qa_obj,
            email_list=email_list,
            trend_insights=trend_insights,
        )

    # Get the text that was printed inside print_full_smm_plan_readable
    human_readable_text = buf.getvalue()

    # Print it back to the notebook so user still sees it
    print(human_readable_text)

    # Save it to a .txt file in Kaggle output dir
    output_path = "/kaggle/working/smm_plan_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(human_readable_text)

    print(f"\n\nâœ… All agents have finished for this client.")
    print(f"ğŸ“� Human-readable plan saved to: {output_path}")


# ğŸ–¥ One-Stop Runner

client_brief_text = '''Client name: FreshBite Cafe
Industry: Cafe & casual dining restaurant
Target audience: Young professionals and students aged 18â€“35 in urban areas
Location: City center area
Tone of voice: Friendly, energetic, slightly humorous
Goals: Increase footfall, promote signature dishes, grow Instagram followers'''

print("\nâœ… Client brief captured. Running full SMM suite...\n")

await run_smm_suite_for_client(client_brief_text, days=7)

