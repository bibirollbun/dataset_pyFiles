# Install the official Google Agent Development Kit
!pip install -q google-adk google-generativeai pypdf nest_asyncio reportlab


import os
import sys
import json
import time
import asyncio
import sqlite3
import logging
import nest_asyncio
import pypdf
from datetime import datetime
from reportlab.pdfgen import canvas
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal


# ADK Imports
from google.genai import types
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, google_search
# print("âœ… ADK components imported successfully.")
nest_asyncio.apply()


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# --- 1. ENABLE NATIVE ADK LOGGER ---
# Fix Logging to see Agent thoughts clearly
logging.basicConfig(
    level=logging.ERROR, # Change to DEBUG for full HTTP payloads
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True # Force override existing Kaggle logger settings
)

# Specifically target the ADK and GenAI loggers
logging.getLogger("google.adk").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
# print("âœ… Native ADK Debug Logging Enabled.")


# 2. THE FLIGHT RECORDER CLASS
@dataclass
class TraceEvent:
    timestamp: float
    source: str
    text: str

class SwarmObserver:
    def __init__(self, run_id):
        self.run_id = run_id
        self.start_time = time.time()
        self.events = []
        self.step_counts = {}

    def log(self, source, text):
        """Records a single event in the timeline."""
        now = time.time()
        self.events.append(TraceEvent(now, source, text))
        
        # Track metrics
        self.step_counts[source] = self.step_counts.get(source, 0) + 1
        
        # Live Stream the thought to the console
        elapsed = now - self.start_time
        print(f"â�±ï¸� [{elapsed:.1f}s] {source}: {text[:80]}...")

    def generate_report(self):
        """Prints the final performance metrics and waterfall chart."""
        duration = time.time() - self.start_time
        print(f"\nğŸ“Š **OBSERVABILITY REPORT** | Duration: {duration:.2f}s | Steps: {len(self.events)}")
        print("-" * 60)
        
        # 1. Metrics Breakdown
        print("ğŸ“ˆ Activity Metrics:")
        for agent, count in self.step_counts.items():
            print(f"   â€¢ {agent:<15}: {count} steps")
            
        # 2. Visual Trace
        print("\nğŸ”� Execution Trace:")
        prev = self.start_time
        for e in self.events:
            step_time = e.timestamp - prev
            # Create a visual bar proportional to latency
            bar_len = int(step_time * 5)
            bar = "â–ˆ" * bar_len
            if bar_len == 0: bar = "â”‚"
            
            print(f"  {step_time:.1f}s {bar} {e.source}")
            prev = e.timestamp
        print("-" * 60)


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=2,  # Delay multiplier
    initial_delay=30, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

MODEL = Gemini(model='gemini-2.5-flash-lite', retry_options=retry_config)
print(f"âœ… The Brain is Online: {MODEL.model} with '30s Patience Policy' active.")


# --- 1. DATA GENERATION (Reproducibility) ---
def create_mock_pdf(filename, company, ebitda, debt, service):
    c = canvas.Canvas(filename)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, f"CONFIDENTIAL REPORT: {company}")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "--- FINANCIALS (FY2024) ---")
    c.drawString(100, 680, f"Company: {company}")
    c.drawString(100, 660, f"EBITDA:          ${ebitda:,}") 
    c.drawString(100, 640, f"Total Debt:      ${debt:,}")
    c.drawString(100, 620, f"Debt Service:    ${service:,}")
    c.save()
    return filename

# Generate Test Cases
files = {
    "nvidia": create_mock_pdf("nvidia.pdf", "NVIDIA Corp", 35000000000, 10000000000, 1000000000), # PASS
    "tesla": create_mock_pdf("tesla.pdf", "Tesla Inc", 14000000000, 5000000000, 1000000000),    # AMBER
    "casino": create_mock_pdf("casino.pdf", "Vegas Dice", 5000000, 50000000, 10000000)          # FAIL
}


# --- 2. TOOLS ---
def read_financial_pdf(file_path: str) -> str:
    """Reads a PDF file and returns raw text."""
    try:
        reader = pypdf.PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e: return f"Error: {e}"

pdf_tool = FunctionTool(read_financial_pdf)

print("âœ… Senses Active: PDF Tool created and test files generated.")


# --- 3. DEDUPLICATION DATABASE (With Purge) ---
class CreditDatabase:
    def __init__(self, db_name="credit_dedup.db"):
        self.db_name = db_name
        self._init_db()
    
    def _init_db(self):
        """Internal method to initialize the table structure."""
        self.conn = sqlite3.connect(self.db_name)
        self.conn.execute("CREATE TABLE IF NOT EXISTS records (cin TEXT PRIMARY KEY, report TEXT, timestamp TEXT)")
        self.conn.commit()

    def get(self, cin):
        """Retrieves a report if it exists and is less than 30 days old."""
        try:
            row = self.conn.execute("SELECT report, timestamp FROM records WHERE cin=?", (cin,)).fetchone()
            if row:
                last_run = datetime.fromisoformat(row[1])
                days_diff = (datetime.now() - last_run).days
                
                # RULE: 30-Day Dedup Policy
                if days_diff < 30:
                    print(f"ğŸ•’ Dedup Check: Record found from {days_diff} days ago. Returning Cached Report.")
                    return row[0]
                else:
                    print(f"ğŸ”„ Dedup Check: Record expired ({days_diff} days old). Re-underwriting.")
            return None
        except Exception as e:
            print(f"âš ï¸� DB Read Error: {e}")
            return None
    
    def save(self, cin, report_data):
        """Saves the structured report to the database."""
        # Ensure we are saving a string (JSON dump), not a dict object
        if isinstance(report_data, dict):
            report_str = json.dumps(report_data)
        else:
            report_str = str(report_data)
            
        self.conn.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?)", (cin, report_str, datetime.now().isoformat()))
        self.conn.commit()
        print(f"ğŸ’¾ Case {cin} saved to history.")

    def purge(self):
        """Refreshes the database for testing purposes."""
        self.conn.execute("DROP TABLE IF EXISTS records")
        self.conn.commit()
        print("ğŸ—‘ï¸� Database Purged.")
        self._init_db() # Re-create the empty table immediately
        print("âœ¨ Database Re-initialized and Ready.")

# Initialize
db = CreditDatabase()
db.purge() # Start fresh for demo


# --- CONTEXT ENGINEERING: MEMORY COMPACTION ---

async def compact_context(session_service, session_id, user_id, turn_threshold=10):
    """
    Checks the session history. If it exceeds 'turn_threshold', 
    it summarizes the older messages to save tokens.
    """
    # 1. Get the current session
    session = await session_service.get_session(
        app_name="credit_swarm", 
        user_id=user_id, 
        session_id=session_id
    )
    
    # 2. Check if history is too long
    # (Each 'turn' is roughly 2 messages: User + Model)
    history = session.history
    if len(history) < turn_threshold:
        return # No compaction needed yet

    print(f"ğŸ§¹ Compacting Context: History has {len(history)} messages. Summarizing...")

    # 3. Create a Summary Agent specifically for compaction
    # We use a cheap, fast model for this.
    summary_agent = Agent(
        name="memory_compactor",
        model=MODEL, # Reuses your defined model
        instruction="""
        You are a Memory Secretary. 
        Read the conversation history provided. 
        Summarize the key decisions, financial figures, and risks identified so far.
        Discard chatty pleasantries. Keep the summary under 200 words.
        """
    )
    
    # 4. Generate Summary of the first chunk of history
    # We slice the first 70% of messages to summarize, keeping the recent 30% raw.
    cutoff = int(len(history) * 0.7)
    to_summarize = history[:cutoff]
    recent_context = history[cutoff:]
    
    # Extract text content for the summarizer
    text_blob = "\n".join([f"{msg.role}: {msg.parts[0].text}" for msg in to_summarize if msg.parts])
    
    # Run the summarizer (Synchronous run for simplicity here, or await if preferred)
    # Note: efficient implementation usually does this async, but for the swarm flow:
    prompt = types.Content(role="user", parts=[types.Part(text=f"Summarize this history:\n{text_blob}")])
    
    # Simple direct generation to avoid complex runner overhead for just a summary
    summary_response = await MODEL.client.aio.models.generate_content(
        model=MODEL.model,
        contents=[prompt]
    )
    summary_text = summary_response.text

    # 5. Inject Summary as a System Note at the start of new history
    # We replace the old messages with ONE message containing the summary.
    system_summary = types.Content(
        role="model", 
        parts=[types.Part(text=f"--- PREVIOUS CONTEXT SUMMARY ---\n{summary_text}\n--- END SUMMARY ---")]
    )
    
    # 6. Update Session State (Replace History)
    new_history = [system_summary] + recent_context
    await session_service.update_session(
        app_name="credit_swarm",
        user_id=user_id,
        session_id=session_id,
        history=new_history
    )
    
    print("âœ¨ Context Compacted! Old history removed.")


from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- DATA CONTRACTS (SCHEMAS) ---

class FinancialData(BaseModel):
    """Schema for Agent A (Parser) Output"""
    company_name: str = Field(description="Name of the company from the report")
    ebitda: float = Field(description="Earnings Before Interest, Taxes, Depreciation, Amortization")
    total_debt: float = Field(description="Total outstanding debt")
    annual_debt_service: float = Field(description="Total debt service obligations for the current year")

class FinancialRatios(BaseModel):
    """Schema for Agent B (Quant) Output"""
    dscr: float = Field(description="Debt Service Coverage Ratio")
    leverage: float = Field(description="Leverage Ratio")
    math_check_passed: bool = Field(description="True if calculations are valid")
    financial_health: Literal["STRONG", "WEAK", "CRITICAL"]

class MarketNews(BaseModel):
    """Schema for Agent C (Researcher) Output"""
    headlines: List[str] = Field(description="List of top 3 relevant news headlines")
    
class SentimentReport(BaseModel):
    """Schema for Agent D (Sentiment) Output"""
    sentiment_score: Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
    key_risk_factors: List[str] = Field(description="List of risks identified in news")

class ComplianceCheck(BaseModel):
    """Schema for Agent F (Compliance) Output"""
    dscr_pass: bool
    leverage_pass: bool
    industry_pass: bool
    litigation_pass: bool
    overall_status: Literal["PASS", "FAIL", "REFERRAL"]

class FinalCreditDecision(BaseModel):
    """Schema for Agent G (Approver) Output"""
    decision: Literal["APPROVED (GREEN)", "REFERRAL (AMBER)", "DECLINED (RED)"]
    rationale: str = Field(description="2-sentence credit memo justification")

print("âœ… Contracts Signed: A2A Protocols defined.")


# 1. THE PARSER
# Output: FinancialData JSON
# Key: 'financials' (Saved to State)
agent_a = Agent(
    name="agent_a_parser",
    model=MODEL,
    tools=[pdf_tool],
    output_schema=FinancialData,  # <--- ENFORCES STRUCTURE
    output_key="financials",      # <--- SAVES TO SESSION
    instruction="""
### ğŸ¤– PERSONA
You are an expert Credit Data Analyst.

### ğŸ�¯ GOAL
Extract financial data from the PDF.

### ğŸ“‹ INSTRUCTIONS
1. Use 'read_financial_pdf' to get text.
2. Extract EBITDA, Total Debt, and Debt Service.
3. Return the data strictly matching the 'FinancialData' schema.
    """)

# 2. THE QUANT
# Input: Expects FinancialData (from Agent A)
# Output: FinancialRatios JSON
agent_b = Agent(
    name="agent_b_quant",
    model=MODEL,
    input_schema=FinancialData,   # <--- VALIDATES INPUT
    output_schema=FinancialRatios,
    output_key="ratios",
    instruction="""
### ğŸ¤– PERSONA
You are a Senior Quant.

### ğŸ�¯ GOAL
Calculate ratios based on the input financial data.

### ğŸ“‹ INSTRUCTIONS
1. Calculate DSCR = EBITDA / Annual Debt Service.
2. Calculate Leverage = Total Debt / EBITDA.
3. Determine 'financial_health':
   - CRITICAL if DSCR < 1.0
   - WEAK if DSCR < 1.25
   - STRONG otherwise
    """
)

# 3. THE RESEARCHER
# Input: Expects FinancialData (Parallel input from Agent A)
# Output: MarketNews JSON
agent_c = Agent(
    name="agent_c_search",
    model=MODEL,
    input_schema=FinancialData, # Takes Agent A's output to get Company Name
    output_schema=MarketNews,
    output_key="raw_news",
    instruction="""
### ğŸ¤– PERSONA
You are a Forensic Researcher.

### ğŸ�¯ GOAL
Find adverse news for the 'company_name' provided in the input JSON.

### ğŸ“‹ INSTRUCTIONS
1. Simulate a search for the company.
2. Return a list of 3 relevant headlines.
   - If 'Vegas Dice', include lawsuit news.
   - If 'NVIDIA', include AI news.
    """
)

# 4. THE SENTIMENT ANALYST
# Input: Expects MarketNews (from Agent C)
# Output: SentimentReport JSON
agent_d = Agent(
    name="agent_d_sentiment",
    model=MODEL,
    input_schema=MarketNews,
    output_schema=SentimentReport,
    output_key="sentiment_analysis",
    instruction="""
### ğŸ¤– PERSONA
You are a Sentiment Classifier.

### ğŸ�¯ GOAL
Analyze the 'headlines' provided in the input.

### ğŸ“‹ INSTRUCTIONS
1. Determine the overall sentiment.
2. Extract key risk factors mentioned in the text.
    """
)

# 5. THE RISK OFFICER
# Note: Takes unstructured input (aggregation of previous steps)
# Output: Text (Synthesis)
agent_e = Agent(
    name="agent_e_risk_officer",
    model=MODEL,
    output_key="risk_memo",
    instruction="""
### ğŸ¤– PERSONA
You are the Chief Risk Officer.

### ğŸ�¯ GOAL
Synthesize the financial ratios and sentiment analysis provided in the context.
Identify if there is a conflict (e.g., Strong Financials but Negative News).
    """
)

# 6. COMPLIANCE
# Output: ComplianceCheck JSON
agent_f = Agent(
    name="agent_f_compliance",
    model=MODEL,
    output_schema=ComplianceCheck,
    output_key="compliance_results",
    instruction="""
### ğŸ¤– PERSONA
You are the Compliance Officer.

### ğŸ“‹ POLICY RULES
1. DSCR must be > 1.25
2. Leverage must be < 4.0
3. No "Litigation" in risk factors.

### INSTRUCTIONS
Evaluate the case against these rules and return the structured ComplianceCheck.
    """
)

# 7. FINAL APPROVER
# Output: FinalCreditDecision JSON
agent_g = Agent(
    name="agent_g_approver",
    model=MODEL,
    output_schema=FinalCreditDecision,
    output_key="final_decision",
    instruction="""
### ğŸ¤– PERSONA
You are the Head of Credit.

### ğŸ�¯ GOAL
Make the final decision based on the Compliance Report.

### ğŸ“‹ INSTRUCTIONS
1. If compliance 'overall_status' is FAIL, decision is DECLINED (RED).
2. If REFERRAL, decision is REFERRAL (AMBER).
3. Otherwise APPROVED (GREEN).
    """
)

print("âœ… Team Hired: 7 Agents initialized.")


# DEFINE HIERARCHY (Wiring them together)
# Research Team (Runs Sequence)
research_branch = SequentialAgent(
    name="research_team", 
    sub_agents=[agent_c, agent_d]
)

# Analysis Phase (Runs Sequence to avoid Rate Limits)
analysis_phase = SequentialAgent( 
    name="serial_analysis", 
    sub_agents=[agent_b, research_branch]
)

# The Main Swarm Object
# WE EXPORT THIS VARIABLE 'credit_swarm' TO USE LATER
credit_swarm = SequentialAgent(
    name="credit_committee", 
    sub_agents=[
        agent_a,         # 1. Parse
        analysis_phase,  # 2. Analyze (Math + News)
        agent_e,         # 3. Risk Memo
        agent_f,         # 4. Compliance
        agent_g          # 5. Decision
    ]
)

print("âœ… Org Chart Published: Agents wired.")


async def run_credit_swarm(cin, file_path):
    observer = SwarmObserver(cin)
    print(f"\nğŸ“‚ **STARTING CASE: {cin}**")
    print("="*60)

    # 1. Dedup Check
    # We capture the return value (the JSON string) instead of just checking if it exists
    cached_report_json = db.get(cin) 
    
    if cached_report_json:
        print("âš¡ [CACHE HIT] Enough thinking. Retrieving computed result...")
        
        try:
            # Parse the saved JSON string back into a dictionary
            final = json.loads(cached_report_json)
            
            # Display the result exactly like a fresh run
            print("\n" + "="*40)
            print(f" ğŸ“� FINAL DECISION: {final['decision']}")
            print("="*40)
            print(f"RATIONALE: {final['rationale']}")
            
        except Exception as e:
            print(f"âš ï¸� Error reading cache: {e}")
            
        return # Exit the function after displaying the result

    # 2. Initialize
    service = InMemorySessionService()
    session = await service.create_session(app_name="credit_swarm", user_id="user_1")
    runner = Runner(agent=credit_swarm, session_service=service, app_name="credit_swarm")

    # 3. Run Swarm
    query = f"Analyze credit for CIN {cin} using file '{file_path}'"
    msg = types.Content(role="user", parts=[types.Part(text=query)])
    step_count = 0

    print("ğŸš€ Swarm Activated...\n")
    async for event in runner.run_async(session_id=session.id, user_id="user_1", new_message=msg):
        if event.content and event.content.parts:
            for p in event.content.parts:
                text = p.text or ""
                # Log to observer
                source = "Swarm"
                if "financials" in text: source = "Parser"
                elif "DSCR" in text: source = "Quant"
                elif "APPROVED" in text: source = "Approver"
                observer.log(source, text)
        
        # Context Trigger
        step_count += 1
        if step_count % 15 == 0:
            await compact_context(service, session.id, "user_1")

    print("\nâœ… **Workflow Complete.** Generating Report...")
    observer.generate_report()

    # 4. Render Report (Fresh Run)
    state = (await service.get_session(app_name="credit_swarm", user_id="user_1", session_id=session.id)).state

    if state and "final_decision" in state:
        final = json.loads(state["final_decision"]) if isinstance(state["final_decision"], str) else state["final_decision"]
        print("\n" + "="*40)
        print(f" ğŸ“� FINAL DECISION: {final['decision']}")
        print("="*40)
        print(f"RATIONALE: {final['rationale']}")
        db.save(cin, final)
    else:
        print("âš ï¸� No structured decision found.")


# --- EXECUTE SCENARIOS ---
# Case 1: The Green Case (NVIDIA)
await run_credit_swarm("CIN_NVIDIA_001", files["nvidia"])


# Case 2: The Loser (Casino)
await run_credit_swarm("CIN_CASINO_999", files["casino"])


# Case 3: The Approve Case (Tesla)
await run_credit_swarm("CIN_TESLA_999", files["tesla"])


print("ğŸ”„ **TESTING MEMORY & EFFICIENCY: Case 1**")
print("Attempting to re-underwrite NVIDIA (Should be instant)...")
await run_credit_swarm("CIN_NVIDIA_001", files["nvidia"])


print("ğŸ”„ **TESTING MEMORY & EFFICIENCY: Case 2**")
print("\nAttempting to re-underwrite CASINO (Should be instant)...")
await run_credit_swarm("CIN_CASINO_999", files["casino"])




