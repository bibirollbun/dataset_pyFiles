# =====================================================
# FinSMS-AI Project with Session Management
# Enhanced with ADK Sessions, State Management, Context Compaction + LOGGING
# =====================================================

import numpy as np
import pandas as pd
import os
import re
import uuid
import requests
from io import StringIO
from typing import List, Optional, Dict, Any
from tqdm import tqdm

# =====================================================
# GLOBAL LOGGING SETUP
# =====================================================

import logging
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)

LOG_FILE_MAIN = "logs/finsms_main.log"
LOG_FILE_AGENTS = "logs/finsms_agents.log"

# Root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE_MAIN, maxBytes=5_000_000, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("FinSMS-Main")

agent_logger = logging.getLogger("FinSMS-Agents")
agent_handler = RotatingFileHandler(LOG_FILE_AGENTS, maxBytes=5_000_000, backupCount=5)
agent_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
agent_logger.addHandler(agent_handler)

logger.info("ğŸ”§ Logging initialized successfully.")

# =====================================================
# Section 1: Setup and Authentication
# =====================================================

from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    logger.info("Gemini API key setup complete.")
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    logger.error(f"Authentication Error: {e}")
    print(f"ğŸ”’ Authentication Error: {e}")

# =====================================================
# Section 2: Import ADK Components with Sessions
# =====================================================

logger.info("Importing ADK components...")

from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, AgentTool, ToolContext, load_memory, preload_memory
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

logger.info("ADK components imported successfully.")
print("âœ… ADK components imported successfully.")

# =====================================================
# Section 3: Configuration and Retry Options
# =====================================================

APP_NAME = "finsms_ai_app"
USER_ID = "default_user"
MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

db_url = "sqlite:///finsms_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)

memory_service = InMemoryMemoryService()

logger.info("Configuration loaded.")
logger.debug(f"Application={APP_NAME}, User={USER_ID}, DB={db_url}")

print("âœ… Configuration complete.")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Database: finsms_agent_data.db")
print(f"   - Memory Service: InMemoryMemoryService (cross-session knowledge)")

# =====================================================
# Section 4: Load Sample SMS Data
# =====================================================

logger.info("Loading sample SMS dataset...")

csv_data = """
id,source,sms_text,date
1,HDFCBK,"INR 1250 debited from A/c ****1234 towards UPI payment at ZOMATO. Ref:239922",2025-01-05 14:20
2,ICICIB,"Rs.5000 credited to your A/c XX8932 via IMPS. Ref:993244",2025-01-06 09:31
3,GPAY,"You paid â‚¹239 to Swiggy using Google Pay. UPI Ref:229388",2025-01-07 12:44
4,PHNPE,"Received â‚¹1000 from Arjun via PhonePe. Ref:553920",2025-01-07 15:22
5,SBICRD,"Your SBI Credit Card ending 5521 is used for Rs 3499 at AMAZON.",2025-01-08 18:10
"""

df = pd.read_csv(StringIO(csv_data))
logger.info(f"Loaded {len(df)} sample SMS records.")

print("âœ… Sample SMS data loaded.")
print(df.head())

# =====================================================
# Section 5: Session State Management Tools
# =====================================================

def save_user_preferences(
    tool_context: ToolContext,
    budget_limit: float = 0.0,
    currency: str = "INR",
    preferred_categories: List[str] = []
) -> Dict[str, Any]:
    
    agent_logger.debug(
        f"[save_user_preferences] budget={budget_limit}, "
        f"currency={currency}, categories={preferred_categories}"
    )
    
    tool_context.state["user:budget_limit"] = budget_limit
    tool_context.state["user:currency"] = currency
    tool_context.state["user:preferred_categories"] = preferred_categories
    tool_context.state["user:preferences_set"] = True
    
    return {
        "status": "success",
        "message": f"Budget limit set to {currency} {budget_limit}",
        "categories": preferred_categories
    }

def retrieve_user_preferences(tool_context: ToolContext) -> Dict[str, Any]:
    
    agent_logger.debug("[retrieve_user_preferences] Fetching preferences")
    
    return {
        "status": "success",
        "budget_limit": tool_context.state.get("user:budget_limit", 0.0),
        "currency": tool_context.state.get("user:currency", "INR"),
        "preferred_categories": tool_context.state.get("user:preferred_categories", []),
        "preferences_set": tool_context.state.get("user:preferences_set", False),
    }

print("âœ… Session state management tools created.")
logger.info("Session state tools loaded.")

# =====================================================
# Section 6: Agent 1 â€” Ingest Agent
# =====================================================

def ingest_sms(id: int, source: str, sms_text: str, date: str) -> dict:
    agent_logger.debug(f"[INGEST] SMS {id}: source={source}")
    return {
        "sms_id": id,
        "source": source,
        "raw_text": sms_text,
        "received_at": date,
        "metadata": {}
    }

ingest_agent = LlmAgent(
    name="ingest_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    You are an ingestion agent for financial SMS data.
    ONLY call ingest_sms().
    """,
    tools=[ingest_sms],
    output_key="ingested_sms"
)

ingest_app = App(name="ingest_app", root_agent=ingest_agent)

ingest_runner = Runner(
    app=ingest_app,
    session_service=session_service
)

logger.info("Ingest agent initialized.")
print("âœ… Ingest Agent with session support created.")

# =====================================================
# Section 7: Agent 2 - Preprocess Agent with Sessions
# =====================================================

def preprocess_sms(
    sms_id: int,
    raw_text: str,
    received_at: str,
    source: str = "",
    metadata: Optional[dict] = None
) -> dict:
    """Preprocess an SMS and extract regex-based hints."""
    
    agent_logger.debug(f"[PREPROCESS] Starting for SMS {sms_id}")

    text = raw_text.strip()

    # Amount extraction
    amount_matches = re.findall(r"(?:INR|Rs\.?|â‚¹)\s?([\d,]+(?:\.\d+)?)", text, flags=re.I)
    amount_candidates = []

    for amt in amount_matches:
        try:
            amount_candidates.append(float(amt.replace(",", "")))
        except:
            agent_logger.warning(f"[PREPROCESS] Failed to parse amount: {amt}")

    # Merchant extraction
    merchant_keywords = re.findall(
        r"(Paytm|PhonePe|GPay|Swiggy|Zomato|Amazon|Flipkart|UPI|IMPS|POS)",
        text,
        flags=re.I
    )

    # OTP detection
    is_otp = bool(re.search(r"\b\d{4,8}\b", text) and "otp" in text.lower())

    # Spam detection
    spam_words = ["win", "offer", "loan", "click", "claim"]
    is_spam = any(w in text.lower() for w in spam_words)

    agent_logger.debug(
        f"[PREPROCESS] SMS {sms_id} processed | "
        f"amounts={amount_candidates}, merchants={merchant_keywords}, "
        f"OTP={is_otp}, spam={is_spam}"
    )

    return {
        "sms_id": sms_id,
        "cleaned_text": text,
        "amount_candidates": amount_candidates,
        "date_candidates": [received_at] if received_at else [],
        "merchant_candidates": list(set(merchant_keywords)),
        "is_otp": is_otp,
        "is_spam": is_spam,
        "source": source,
        "metadata": metadata or {}
    }

preprocess_agent = LlmAgent(
    name="preprocess_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    ONLY call preprocess_sms().
    """,
    tools=[preprocess_sms],
    output_key="preprocessed_sms"
)

preprocess_app = App(name="preprocess_app", root_agent=preprocess_agent)

preprocess_runner = Runner(
    app=preprocess_app,
    session_service=session_service
)

logger.info("Preprocess agent initialized.")
print("âœ… Preprocess Agent with session support created.")

# =====================================================
# Section 8: HuggingFace Integration Tools
# =====================================================

HF_API_KEY = "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"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def hf_category_cloud(text: str, labels: List[str]) -> dict:
    agent_logger.debug(f"[HF-CATEGORY] text={text[:25]}..., labels={labels}")
    url = "https://router.huggingface.co/hf-inference/facebook/bart-large-mnli"
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    
    resp = requests.post(url, headers=HF_HEADERS, json=payload)
    try:
        data = resp.json()
        agent_logger.debug(f"[HF-CATEGORY] Response OK")
        return data
    except:
        raw = resp.text
        agent_logger.error("[HF-CATEGORY] Non-JSON Response")
        return {"error": "Non-JSON", "raw": raw}

def hf_similarity_cloud(a: str, b: str) -> dict:
    agent_logger.debug(f"[HF-SIM] Comparing '{a[:25]}...' with '{b[:25]}...'")
    url = "https://router.huggingface.co/hf-inference/sentence-transformers/all-MiniLM-L6-v2"
    
    r1 = requests.post(url, headers=HF_HEADERS, json={"inputs": a})
    r2 = requests.post(url, headers=HF_HEADERS, json={"inputs": b})
    
    try:
        e1 = np.array(r1.json()[0][0])
        e2 = np.array(r2.json()[0][0])
        score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        agent_logger.debug(f"[HF-SIM] Score={score}")
        return {"score": score}
    except:
        agent_logger.error("[HF-SIM] Non-JSON response")
        return {"error": "Non-JSON"}

def hf_ner_cloud(text: str) -> dict:
    agent_logger.debug(f"[HF-NER] text={text[:30]}...")
    url = "https://router.huggingface.co/hf-inference/dslim/bert-base-NER"
    resp = requests.post(url, headers=HF_HEADERS, json={"inputs": text})
    try:
        return resp.json()
    except:
        agent_logger.error("[HF-NER] Non-JSON")
        return {"error": "Non-JSON"}

def hf_spam_cloud(text: str) -> dict:
    agent_logger.debug(f"[HF-SPAM] text={text[:30]}...")
    url = "https://router.huggingface.co/hf-inference/mukeshbankar/spam_sms_detection"
    resp = requests.post(url, headers=HF_HEADERS, json={"inputs": text})
    try:
        return resp.json()
    except:
        agent_logger.error("[HF-SPAM] Non-JSON")
        return {"error": "Non-JSON"}

logger.info("HuggingFace tools initialized.")
print("âœ… HuggingFace integration tools created.")

# =====================================================
# Section 9: Agent 3 - Extract & Classify with Sessions
# =====================================================

def extract_and_classify(
    sms_id: int,
    cleaned_text: str,
    amount_candidates: List[float],
    merchant_candidates: List[str],
    date_candidates: List[str],
    source: str,
    is_otp: bool,
    is_spam: bool,
    amount: Optional[float] = None,
    currency: Optional[str] = None,
    merchant: Optional[str] = None,
    transaction_type: Optional[str] = None,
    debit_or_credit: Optional[str] = None,
    category: Optional[str] = None,
    timestamp: Optional[str] = None,
    confidence: Optional[float] = None,
    explanation: Optional[str] = None,
    metadata: Optional[dict] = None
) -> dict:

    agent_logger.debug(
        f"[EXTRACT] SMS {sms_id} | merchant={merchant} | amount={amount} | "
        f"type={transaction_type} | category={category}"
    )

    return {
        "sms_id": sms_id,
        "amount": amount,
        "currency": currency,
        "merchant": merchant,
        "transaction_type": transaction_type,
        "debit_or_credit": debit_or_credit,
        "category": category,
        "timestamp": timestamp,
        "confidence": confidence,
        "explanation": explanation,
        "raw": {
            "cleaned_text": cleaned_text,
            "amount_candidates": amount_candidates,
            "merchant_candidates": merchant_candidates,
            "date_candidates": date_candidates,
            "source": source,
            "is_otp": is_otp,
            "is_spam": is_spam,
            "metadata": metadata or {}
        }
    }

extract_classify_agent = LlmAgent(
    name="extract_classify_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="ONLY call extract_and_classify() with correct fields.",
    tools=[extract_and_classify],
    output_key="final_extracted"
)

extract_app = App(
    name="extract_classify_app",
    root_agent=extract_classify_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,
        overlap_size=1
    )
)

extract_runner = Runner(
    app=extract_app,
    session_service=session_service
)

logger.info("Extract agent initialized.")
print("âœ… Extract/Classify Agent with sessions and context compaction created.")

# =====================================================
# Section 10: Agent 4 - Summarization with Sessions
# =====================================================

def summarize_transactions(
    sms_ids: List[int],
    amounts: List[float],
    categories: List[str],
    merchants: List[str],
    debit_or_credit: List[str],
    timestamps: List[str],
    metadata: Optional[dict] = None
) -> dict:

    agent_logger.debug(
        f"[SUMMARY] Processing {len(sms_ids)} transactions"
    )

    df_summary = pd.DataFrame({
        "sms_id": sms_ids,
        "amount": amounts,
        "category": categories,
        "merchant": merchants,
        "direction": debit_or_credit,
        "timestamp": timestamps
    })

    df_summary["amount"] = pd.to_numeric(df_summary["amount"], errors="coerce").fillna(0)

    total_spent = df_summary[df_summary["direction"] == "DEBIT"]["amount"].sum()
    total_received = df_summary[df_summary["direction"] == "CREDIT"]["amount"].sum()

    agent_logger.debug(
        f"[SUMMARY] Spent={total_spent}, Received={total_received}"
    )

    return {
        "total_spent": float(total_spent),
        "total_received": float(total_received),
        "category_totals": df_summary.groupby("category")["amount"].sum().to_dict(),
        "merchant_totals": df_summary.groupby("merchant")["amount"].sum().to_dict(),
        "transaction_count": len(df_summary),
        "metadata": metadata or {}
    }

summarization_agent = LlmAgent(
    name="summarization_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="ONLY call summarize_transactions(). Use memory after tool execution.",
    tools=[summarize_transactions, save_user_preferences, retrieve_user_preferences, load_memory],
    output_key="monthly_summary"
)

summarization_app = App(name="summarization_app", root_agent=summarization_agent)

summarization_runner = Runner(
    app=summarization_app,
    session_service=session_service,
    memory_service=memory_service
)

logger.info("Summarization agent initialized.")
print("âœ… Summarization Agent with session and memory support created.")

# =====================================================
# Section 11: Helper Functions for Session Management
# =====================================================

async def run_agent_with_session(
    runner_instance: Runner,
    user_query: str,
    session_id: str
) -> list:
    """
    Run an agent with session management.
    """

    app_name = runner_instance.app.name

    logger.debug(f"[SESSION] Accessing session: {session_id} for app={app_name}")

    try:
        session = await session_service.get_session(
            app_name=app_name,
            user_id=USER_ID,
            session_id=session_id
        )

        if session is None:
            raise ValueError("Session not found")

    except Exception as e:
        logger.warning(f"[SESSION] Creating new session: {session_id} due to {e}")

        session = await session_service.create_session(
            app_name=app_name,
            user_id=USER_ID,
            session_id=session_id
        )

    # Format prompt
    query_content = types.Content(
        role="user",
        parts=[types.Part(text=user_query)]
    )

    events = []
    logger.debug(f"[SESSION] Running agent with prompt: {user_query[:80]}...")

    async for event in runner_instance.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=query_content
    ):
        events.append(event)

    logger.debug(f"[SESSION] Received {len(events)} events.")
    return events


def parse_tool_output(events):
    """Extract tool output from ADK events."""
    logger.debug(f"[PARSE] Parsing {len(events)} ADK events")

    for event in events:
        if (
            getattr(event, "content", None)
            and getattr(event.content, "parts", None)
            and len(event.content.parts) > 0
            and getattr(event.content.parts[0], "function_response", None)
        ):
            result = event.content.parts[0].function_response.response
            logger.debug(f"[PARSE] Tool output extracted successfully")
            return result

    logger.warning("[PARSE] No tool output found in events")
    return None


async def auto_save_to_memory(callback_context):
    """Auto-save memory after agent turn."""
    try:
        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
        )
        logger.info("ğŸ’¾ Memory saved successfully")
    except Exception as e:
        logger.error(f"[MEMORY] Save failed: {e}")

print("âœ… Session helper functions defined.")
logger.info("Session helpers loaded.")

# =====================================================
# Section 12: Complete Pipeline Execution with Sessions
# =====================================================

logger.info("=== Starting FinSMS Full Pipeline ===")
print("\n" + "="*60)
print("STARTING FINSMS-AI PIPELINE WITH SESSION MANAGEMENT")
print("="*60)

# ---------------------------------------
# STEP 1 â€” INGEST
# ---------------------------------------

logger.info("ğŸ“¥ Step 1: Ingesting SMS Data...")
print("\nğŸ“¥ Step 1: Ingesting SMS Data...")

ingested = []
ingest_session_id = f"ingest_session_{uuid.uuid4().hex[:8]}"

ingest_memory_runner = InMemoryRunner(agent=ingest_agent)

rows = df.head(5).to_dict(orient="records")

for idx, row in enumerate(rows):
    logger.debug(f"[INGEST] Processing row {idx+1}: {row}")

    prompt = f"""
Normalize this SMS record using ingest_sms tool:
id: {row['id']}
source: "{row['source']}"
sms_text: "{row['sms_text']}"
date: "{row['date']}"
"""

    response = await ingest_memory_runner.run_debug(prompt)
    logger.debug(f"[INGEST] Raw agent response received")

    # Extract tool result
    tool_result = None
    for event in response:
        if (
            event.content.parts
            and event.content.parts[0].function_response
        ):
            tool_result = event.content.parts[0].function_response.response
            break

    if tool_result:
        logger.info(f"[INGEST] SMS {idx+1} ingested successfully")
        ingested.append(tool_result)
    else:
        logger.error(f"[INGEST] SMS {idx+1} FAILED to ingest")

    print(f"  âœ“ Ingested SMS {idx+1}/{len(rows)}")

df_ingested = pd.DataFrame(ingested)
logger.info(f"[INGEST] Completed ingestion | {len(df_ingested)} records")
print(f"âœ… Ingestion complete: {len(df_ingested)} records")

# ---------------------------------------
# STEP 2 â€” PREPROCESS
# ---------------------------------------

logger.info("ğŸ”§ Step 2: Preprocessing SMS Data...")
print("\nğŸ”§ Step 2: Preprocessing SMS Data...")

preprocessed_records = []
preprocess_memory_runner = InMemoryRunner(agent=preprocess_agent)

for idx, row in enumerate(df_ingested.to_dict(orient="records")):
    logger.debug(f"[PREPROCESS] Processing SMS {row['sms_id']}")

    prompt = f"""
Preprocess this SMS using preprocess_sms tool.

sms_id: {row['sms_id']}
raw_text: "{row['raw_text']}"
received_at: "{row['received_at']}"
source: "{row['source']}"
metadata: {row['metadata']}
"""

    response = await preprocess_memory_runner.run_debug(prompt)

    # Extract tool output
    tool_output = None
    for event in response:
        if (
            event.content.parts
            and event.content.parts[0].function_response
        ):
            tool_output = event.content.parts[0].function_response.response
            break

    if tool_output:
        logger.info(f"[PREPROCESS] SMS {idx+1} preprocessed successfully")
        preprocessed_records.append(tool_output)
    else:
        logger.error(f"[PREPROCESS] SMS {idx+1} FAILED")

    print(f"  âœ“ Preprocessed SMS {idx+1}/{len(df_ingested)}")

df_preprocessed = pd.DataFrame(preprocessed_records)
logger.info(f"[PREPROCESS] Completed preprocessing | {len(df_preprocessed)} records")
print(f"âœ… Preprocessing complete: {len(df_preprocessed)} records")

# ---------------------------------------
# STEP 3 â€” EXTRACT & CLASSIFY
# ---------------------------------------

logger.info("ğŸ�¯ Step 3: Extracting & Classifying...")
print("\nğŸ�¯ Step 3: Extracting and Classifying Transactions...")

final_outputs = []

extract_memory_runner = InMemoryRunner(agent=extract_classify_agent)

for idx, row in enumerate(df_preprocessed.to_dict(orient="records")):
    logger.debug(f"[EXTRACT] SMS {row['sms_id']} starting classification.")

    prompt = f"""
Extract and classify this SMS:
SMS ID: {row['sms_id']}
Text: "{row['cleaned_text']}"
Amounts: {row['amount_candidates']}
Merchants: {row['merchant_candidates']}
Date: {row['date_candidates']}
Source: {row['source']}
is_otp: {row['is_otp']}
is_spam: {row['is_spam']}
"""

    response = await extract_memory_runner.run_debug(prompt)

    result = None
    for event in response:
        if (
            event.content.parts
            and event.content.parts[0].function_response
        ):
            result = event.content.parts[0].function_response.response
            break

    if result:
        final_outputs.append(result)
        logger.info(f"[EXTRACT] SMS {idx+1} classified successfully")
        print(f"  âœ“ Extracted SMS {idx+1}/{len(df_preprocessed)}")
    else:
        logger.error(f"[EXTRACT] FAILED SMS {idx+1}")
        print(f"  âœ— Failed to extract SMS {idx+1}/{len(df_preprocessed)}")

# Build dataframe
df_extracted = pd.json_normalize(final_outputs)
logger.info(f"[EXTRACT] Completed extraction | {len(df_extracted)} records")

print(f"âœ… Extraction complete: {len(df_extracted)} records")
print("\nğŸ“‹ Sample extracted row:")
if len(df_extracted) > 0:
    print(df_extracted.iloc[0].to_dict())

# ---------------------------------------
# STEP 4 â€” SUMMARY
# ---------------------------------------

logger.info("ğŸ“Š Step 4: Generating Summary...")
print("\nğŸ“Š Step 4: Generating Financial Summary...")

summarization_memory_runner = InMemoryRunner(agent=summarization_agent)

if len(df_extracted) == 0:
    logger.warning("[SUMMARY] No extracted transactions available")
    summary_output = None
    print("â�Œ No transactions to summarize")

else:
    sms_ids = df_extracted["sms_id"].tolist()
    amounts = df_extracted["amount"].tolist()
    categories = df_extracted["category"].fillna("Uncategorized").tolist()
    merchants = df_extracted["merchant"].tolist()
    debit_or_credit = df_extracted["debit_or_credit"].tolist()
    timestamps = df_extracted["timestamp"].tolist()

    summary_prompt = f"""
Call summarize_transactions() with:
sms_ids: {sms_ids}
amounts: {amounts}
categories: {categories}
merchants: {merchants}
debit_or_credit: {debit_or_credit}
timestamps: {timestamps}
metadata: {{}}
"""

    logger.debug(f"[SUMMARY] Sending summary prompt")
    response = await summarization_memory_runner.run_debug(summary_prompt)

    summary_output = None
    for event in response:
        if (
            event.content.parts
            and event.content.parts[0].function_response
        ):
            summary_output = event.content.parts[0].function_response.response
            break

    if summary_output:
        logger.info(f"[SUMMARY] Summary generated successfully")
        print("\nâœ… Summary Generated:")
        print(summary_output)
    else:
        logger.error("[SUMMARY] Summary generation FAILED")
        print("â�Œ Failed to generate summary")


# =====================================================
# PART 4 â€” Excel export, final display, session inspection, shutdown logs
# =====================================================

# =====================================================
# Section 13: Display Final Results (with logging)
# =====================================================

logger.info("Displaying final results")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

if len(df_extracted) > 0:
    logger.info(f"[RESULTS] Displaying {len(df_extracted)} extracted transactions")
    print("\nğŸ“‹ Extracted Transactions:")

    display_cols = []
    for col in ['sms_id', 'amount', 'merchant', 'category', 'debit_or_credit', 'timestamp']:
        if col in df_extracted.columns:
            display_cols.append(col)

    if display_cols:
        print(df_extracted[display_cols].to_string())
    else:
        print(df_extracted.to_string())

    if summary_output:
        logger.info("[RESULTS] Printing summary output")
        print("\nğŸ’° Financial Summary:")
        print(f"   Total Spent: â‚¹{summary_output['total_spent']:.2f}")
        print(f"   Total Received: â‚¹{summary_output['total_received']:.2f}")
        print(f"   Net: â‚¹{summary_output['total_received'] - summary_output['total_spent']:.2f}")

        print("\nğŸ“Š Category Breakdown:")
        for category, amount in summary_output['category_totals'].items():
            print(f"   {category}: â‚¹{amount:.2f}")

        print("\nğŸ�ª Merchant Breakdown:")
        for merchant, amount in summary_output['merchant_totals'].items():
            print(f"   {merchant}: â‚¹{amount:.2f}")
    else:
        logger.warning("[RESULTS] Summary output not available")
        print("\nâš ï¸�  Summary not available")
else:
    logger.warning("[RESULTS] No transactions were extracted")
    print("\nâ�Œ No transactions were extracted")

print("\n" + "="*60)
print("âœ… PIPELINE COMPLETE")
print("="*60)

# =====================================================
# Section 14: Export to Excel (with logging)
# =====================================================

logger.info("Exporting results to Excel/CSV")

print("\nğŸ“¤ Exporting Results to Excel...")

try:
    excel_filename = 'FinSMS_Analysis_Report.xlsx'

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:

        # Sheet 1: Transactions Detail
        if len(df_extracted) > 0:
            export_df = df_extracted.copy()

            cols_to_export = []
            col_mapping = {
                'sms_id': 'Transaction ID',
                'timestamp': 'Date & Time',
                'merchant': 'Merchant',
                'amount': 'Amount (â‚¹)',
                'debit_or_credit': 'Type',
                'category': 'Category',
                'transaction_type': 'Payment Method',
                'explanation': 'Description'
            }

            for old_col, new_col in col_mapping.items():
                if old_col in export_df.columns:
                    cols_to_export.append(old_col)

            if cols_to_export:
                export_df = export_df[cols_to_export].rename(columns=col_mapping)
                export_df.to_excel(writer, sheet_name='Transactions', index=False)
                logger.debug(f"[EXPORT] Transactions sheet written with {len(export_df)} rows")

        # Sheet 2: Summary Statistics
        if summary_output:
            summary_data = {
                'Metric': [
                    'Total Spent',
                    'Total Received',
                    'Net Amount',
                    'Total Transactions'
                ],
                'Value (â‚¹)': [
                    f"â‚¹{summary_output['total_spent']:.2f}",
                    f"â‚¹{summary_output['total_received']:.2f}",
                    f"â‚¹{summary_output['total_received'] - summary_output['total_spent']:.2f}",
                    summary_output['transaction_count']
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            logger.debug("[EXPORT] Summary sheet written")

            # Sheet 3: Category Breakdown
            if summary_output.get('category_totals'):
                category_data = {
                    'Category': list(summary_output['category_totals'].keys()),
                    'Amount (â‚¹)': [float(v) for v in summary_output['category_totals'].values()]
                }
                df_categories = pd.DataFrame(category_data)
                df_categories = df_categories.sort_values('Amount (â‚¹)', ascending=False)
                df_categories.to_excel(writer, sheet_name='Category Breakdown', index=False)
                logger.debug("[EXPORT] Category Breakdown sheet written")

            # Sheet 4: Merchant Breakdown
            if summary_output.get('merchant_totals'):
                merchant_data = {
                    'Merchant': list(summary_output['merchant_totals'].keys()),
                    'Amount (â‚¹)': [float(v) for v in summary_output['merchant_totals'].values()]
                }
                df_merchants = pd.DataFrame(merchant_data)
                df_merchants = df_merchants.sort_values('Amount (â‚¹)', ascending=False)
                df_merchants.to_excel(writer, sheet_name='Merchant Breakdown', index=False)
                logger.debug("[EXPORT] Merchant Breakdown sheet written")

        # Sheet 5: Raw SMS Data
        try:
            df.to_excel(writer, sheet_name='Raw SMS Data', index=False)
            logger.debug("[EXPORT] Raw SMS Data sheet written")
        except Exception as e_raw:
            logger.warning(f"[EXPORT] Failed to write Raw SMS Data sheet: {e_raw}")

    logger.info(f"[EXPORT] Excel report created: {excel_filename}")
    print(f"âœ… Excel report created: {excel_filename}")

    # File size and path
    try:
        file_size_kb = os.path.getsize(excel_filename) / 1024
        abs_path = os.path.abspath(excel_filename)
        logger.info(f"[EXPORT] File size: {file_size_kb:.2f} KB, Path: {abs_path}")
        print(f"\nğŸ“� File size: {file_size_kb:.2f} KB")
        print(f"ğŸ“� Location: {abs_path}")
    except Exception as e_stat:
        logger.warning(f"[EXPORT] Could not stat file: {e_stat}")

except Exception as e:
    logger.error(f"[EXPORT] Error creating Excel file: {e}")
    print(f"â�Œ Error creating Excel file: {e}")
    print("Falling back to CSV export...")

    # Fallback to CSV
    try:
        csv_filename = 'FinSMS_Transactions.csv'
        if len(df_extracted) > 0:
            df_extracted.to_csv(csv_filename, index=False)
        else:
            # When df_extracted empty, still export raw data
            df.to_csv(csv_filename, index=False)
        logger.info(f"[EXPORT] CSV export created: {csv_filename}")
        print(f"âœ… CSV file created: {csv_filename}")
    except Exception as csv_error:
        logger.error(f"[EXPORT] CSV export failed: {csv_error}")
        print(f"â�Œ CSV export also failed: {csv_error}")

# =====================================================
# Section 15: Session Inspection (Optional)
# =====================================================

logger.info("Session inspection note: using InMemoryRunner for quick runs.")
print("\nğŸ”� Session Management Note:")
print("   This version uses InMemoryRunner for simplicity.")
print("   Sessions are maintained within each agent's memory during execution.\n")
print("   To enable full persistence, you can:")
print("   1. Use Runner with DatabaseSessionService (session_service is configured).")
print("   2. Manually save df_extracted and summary_output to a DB table/file.")
print("   3. Resume processing from saved checkpoints (store session_id).")

logger.info("FinSMS pipeline execution finished successfully.")
print("\nâœ… All Done! Pipeline executed successfully with agent memory and logging.")


