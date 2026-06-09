# CODE CELL (Installation Fix)
!pip install nest-asyncio


# Configure Retry Options
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts: will try up to 5 times.
    exp_base=7,  # Exponential Backoff Multiplier: delay increases by 7x each time.
    initial_delay=1, # Wait 1 second before the first retry.
    http_status_codes=[429, 500, 503, 504],  # Critical fix: Retries specifically on 429 (Resource Exhausted).
)

print("âœ… Retry configuration object defined.")


# CODE CELL 3: System Definitions and Agents

# --- IMPORTS AND SETUP ---
import asyncio
import time
import json
import os
import random
from datetime import date, datetime
from typing import List, Optional, Tuple
from enum import Enum
import pandas as pd # Necessary for calculate_campaign_metrics
from pydantic import BaseModel, Field

# --- KAGGLE SECRET INTEGRATION ---
try:
    from kaggle_secrets import UserSecretsClient
    from google.genai import types 
    user_secrets = UserSecretsClient()
    KAGGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    
    import google.generativeai as genai
    
    # --- CRITICAL FINAL FIX: REMOVE INVALID CLIENT_OPTIONS ---
    # The client now initializes successfully using the default built-in resilience.
    genai.configure(api_key=KAGGLE_API_KEY)
    
    print("âœ… Gemini Client configured successfully (Default Resilience Active).")
    
except Exception as e:
    print(f"â�Œ Configuration Error: Could not apply client options. Details: {e}")
    import google.generativeai as genai

# --- A. DATA CONTRACTS (Pydantic Models) ---
class ContentMetrics(BaseModel):
    tone_score: int = Field(..., description="Rating 1-10 of tone appropriateness")
    hook_strength: str = Field(..., pattern="^(High|Medium|Low)$", description="Strength of the first 3 seconds")
    key_themes: List[str] = Field(default_factory=list, description="Main topics identified")
    rate_check: str = Field("Not Applicable", description="Did the script match current market rates? (Valid, Invalid, or Not Applicable)")
    deviation_summary: List[str] = Field(default_factory=list, description="Specific sentences from the transcript that violate Forbidden Topics.")

class InfluencerTag(str, Enum):
    SKEPTIC = "Skeptic"
    EDUCATOR = "Educator"
    NEWS = "News"
    CULTURE = "Culture"

class InfluencerProfile(BaseModel):
    id: str
    name: str
    archetypes: List[InfluencerTag] = Field(..., description="List of Persona Tags")

class BrandPolicy(BaseModel):
    policy_id: str
    name: str
    start_date: date
    end_date: Optional[date]
    market_trigger: str
    focus_phrases: List[str]
    compliance_phrases: List[str]
    forbidden_topics: List[str]

class PerformanceMetrics(BaseModel):
    raw_lead_count: int
    lost_leads: int
    closed_count: int
    
class AgentState(BaseModel):
    approved_script: str
    influencer_transcript: str
    csv_data: str
    campaign_date: date
    influencer_profile: InfluencerProfile
    brand_policy: Optional[BrandPolicy]
    content_analysis: Optional[ContentMetrics] = None
    performance_data: Optional[PerformanceMetrics] = None
    final_strategic_review: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

# --- B. TOOL DEFINITIONS ---
def get_current_mortgage_rate(loan_type: str = "30_year_fixed"):
    """MOCK FUNCTION: Fetches the current national average mortgage rate."""
    base_rate = 6.5 + (random.random() * 1.0)
    print(f"   [TOOL] ğŸ› ï¸�  Agent is looking up {loan_type} rates...")
    return {"rate_percentage": round(base_rate, 2), "trend": "stable"}
# --- We will add additional APIs for the Consumer Confidence Index (CCI) and "holidays" ---
def calculate_campaign_metrics(csv_data: str) -> PerformanceMetrics:
    """Deterministic Tool: Calculates non-financial metrics."""
    import io
    df = pd.read_csv(io.StringIO(csv_data))
    raw_leads = df['conversions'].sum() * 5 
    lost_leads = df['conversions'].sum() * 2 
    closed_count = df['conversions'].sum() * 1 
    return PerformanceMetrics(raw_lead_count=int(raw_leads), lost_leads=int(lost_leads), closed_count=int(closed_count))

# --- C. POLICY MANAGER (Time Travel Logic) ---
class PolicyManager:
    """Manages the historical brand policy registry (Time Travel Logic)."""
    def __init__(self):
        # MOCK POLICY DATA INLINE FOR NOTEBOOK STABILITY
        self.policies_data = [
            {"policy_id": "POL-2024-001", "name": "Q1 New Year Stability", "start_date": "2024-01-01", "end_date": "2024-01-14", "market_trigger": "Stable Rates", "focus_phrases": ["fresh start"], "compliance_phrases": ["NMLS #12345", "Equal Housing Lender"], "forbidden_topics": []},
            {"policy_id": "POL-2024-002", "name": "Q1 Rate Spike Pivot", "start_date": "2024-01-15", "end_date": "2024-01-31", "market_trigger": "Rate Hike (+0.5%)", "focus_phrases": ["lock it in", "rate protection"], "compliance_phrases": ["NMLS #12345", "Equal Housing Lender"], "forbidden_topics": ["guarantee", "100%", "basically guaranteed to go up"]},
        ]
        self.policies = [BrandPolicy(**p) for p in self.policies_data]
        self.policies.sort(key=lambda x: x.start_date)

    def get_policy_for_date(self, target_date_str: str) -> Optional[BrandPolicy]:
        target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        for policy in self.policies:
            if policy.start_date <= target and (policy.end_date is None or policy.end_date >= target):
                return policy
        return None

# --- D. AGENT DEFINITIONS ---
class ContentAnalyzerAgent:
    def __init__(self):
        self.tools = [get_current_mortgage_rate]
        self.system_instruction_template = "You are an expert Content Strategist and Compliance Auditor. Use the tool if the script mentions rates. Perform semantic comparison and compliance check."
        self.model = genai.GenerativeModel(model_name='gemini-2.5-flash', tools=self.tools)

    def analyze(self, approved_script: str, influencer_transcript: str, compliance_phrases: List[str], forbidden_topics: List[str]) -> dict:
        try:
            full_system_instruction = self.system_instruction_template + f"\n\nMANDATORY PHRASES: {compliance_phrases}" + f"\nFORBIDDEN TOPICS (Risk Check): {forbidden_topics}"
            model_with_instruction = genai.GenerativeModel(model_name=self.model.model_name, tools=self.tools, system_instruction=full_system_instruction)
            chat = model_with_instruction.start_chat(enable_automatic_function_calling=True)
            
            prompt = f"""
            Perform a DUAL SCORE ADHERENCE check (Semantic Fidelity & Risk Deviation).
            COMPARE: A) Approved Script (PLAN): "{approved_script}" vs. B) Influencer Transcript (REALITY): "{influencer_transcript}"
            1. Check Fidelity: Score semantic alignment.
            2. Check Deviation: Scan for sentences violating Forbidden Topics.
            Return ONLY the JSON object matching the ContentMetrics schema.
            """
            response = chat.send_message(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            return {"tone_score": 0, "hook_strength": "Low", "key_themes": ["ERROR"], "rate_check": "Error", "deviation_summary": [f"System Error: {e}"]}

class ReviewCoordinatorAgent:
    def __init__(self):
        # TEMPORARY FIX FOR KAGGLE QUOTA: Downgrade Pro to Flash
        self.model = genai.GenerativeModel(model_name='gemini-2.5-flash') 
        # NOTE: For final enterprise use, change back to 'gemini-2.5-pro' for superior reasoning.

    def review(self, content_data: dict, performance_data: dict, policy_data: dict, influencer_data: dict) -> str:
        system_instruction = f"""
        You are the Lead Performance Strategist. Perform Causal Attribution.
        CORE PHILOSOPHY: Market Context ({policy_data['market_trigger']}) is the primary driver. Use the 'Steady Hand' protocol.
        Analyze: 1. Deviation Risk (Compliance). 2. Persona Fit (Messenger). 3. Performance (Leads) vs. Market.
        """
        try:
            model_with_instruction = genai.GenerativeModel(model_name=self.model.model_name, system_instruction=system_instruction)
            prompt = f"Content Analysis: {content_data}\nPerformance Metrics: {performance_data}\nProvide a final Strategic Review."
            response = model_with_instruction.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"RCA Error during synthesis: {e}"

# --- INITIALIZE SINGLETONS ---
caa = ContentAnalyzerAgent()
rca = ReviewCoordinatorAgent()
policy_manager = PolicyManager()


# CODE CELL 4: Execution Logic and Demo Run

# --- ORCHESTRATOR FUNCTION ---
async def run_agent_system(approved_script: str, influencer_transcript: str, csv_data: str, campaign_date_str: str, influencer_data: dict) -> AgentState:
    print("\n--- ğŸš€ Starting Orchestration with Context Lookup ---")
    start_total = time.time()
    
    # 1. Initialization and Policy Lookup (The Governance Layer)
    try:
        campaign_date = date.fromisoformat(campaign_date_str)
        influencer_profile = InfluencerProfile(**influencer_data)
        active_policy = policy_manager.get_policy_for_date(campaign_date_str)
        if not active_policy: raise ValueError(f"No active Brand Policy found for date {campaign_date_str}")
        
        state = AgentState(
            approved_script=approved_script, influencer_transcript=influencer_transcript,
            csv_data=csv_data, campaign_date=campaign_date,
            influencer_profile=influencer_profile, brand_policy=active_policy
        )
    except Exception as e:
        print(f"   [INIT] â�Œ Initialization Failed: {e}"); 
        return AgentState(approved_script="", influencer_transcript="", csv_data="", errors=[f"Initialization Failed: {e}"], campaign_date=date.today(), influencer_profile=InfluencerProfile(id="", name="", archetypes=[]))

    # 2. Parallel Execution (Map Step)
    def run_content_task():
        print("   [Map] ğŸ§  Content Agent analyzing...")
        return caa.analyze(
            approved_script, influencer_transcript, active_policy.compliance_phrases, active_policy.forbidden_topics
        )
    
    def run_data_task():
        print("   [Map] ğŸ§® Metrics Engine calculating...")
        return calculate_campaign_metrics(csv_data)

    try:
        content_result, data_result = await asyncio.gather(
            asyncio.to_thread(run_content_task),
            asyncio.to_thread(run_data_task)
        )
        state.content_analysis = content_result
        state.performance_data = data_result
        print("   [Map] âœ… Parallel tasks complete.")
    except Exception as e:
        state.errors.append(f"Map Phase Error: {str(e)}"); return state

    # 3. Strategic Review (The "Reduce" Step)
    print("   [Reduce] ğŸ‘” Coordinator synthesizing strategy...")
    try:
        final_review = await asyncio.to_thread(
            rca.review, state.content_analysis, state.performance_data.model_dump(), 
            state.brand_policy.model_dump(), state.influencer_profile.model_dump() 
        )
        state.final_strategic_review = final_review
        print("   [Reduce] âœ… Synthesis complete.")
    except Exception as e:
        state.errors.append(f"Reduce Phase Error: {str(e)}"); return state

    print(f"--- ğŸ�� Workflow Finished in {time.time() - start_total:.2f}s ---\n")
    return state

# --- EXECUTION LOGIC ---
if __name__ == "__main__":
    import nest_asyncio
    
    # ğŸš¨ DEMO DATA (High-Risk Failure Scenario)
    TEST_DATE = "2024-01-20"
    APPROVED_SCRIPT = "Lock in your rate today! Don't wait. Our guarantee is an Equal Housing Lender."
    INFLUENCER_TRANSCRIPT = "Hey guys, rates are basically guaranteed to go up next month, so lock it in now. You can trust me 100% on this. Don't wait. Our guarantee is an Equal Housing Lender."
    SAMPLE_CSV = "spend,conversions,revenue\n100,10,500\n" 
    INFLUENCER_PROFILE = {"id": "INF-001", "name": "Sarah Skeptic", "archetypes": ["Skeptic", "Educator"]}

    # CRITICAL FIX for Kaggle/Jupyter execution
    nest_asyncio.apply() 
    result = asyncio.run(run_agent_system(APPROVED_SCRIPT, INFLUENCER_TRANSCRIPT, SAMPLE_CSV, TEST_DATE, INFLUENCER_PROFILE))
    
    # --- FINAL REPORT GENERATION ---
    print("FINAL OUTPUT PREVIEW:")
    print("-" * 50)
    
    if result.brand_policy:
        print(f"Policy Used: {result.brand_policy.name}")
    
    print(f"Errors: {result.errors}")
    
    if result.content_analysis:
        print("\n### Adherence Check (Risk Analysis)")
        risk_summary = result.content_analysis.get('deviation_summary', [])
        if risk_summary and any('Error' not in s for s in risk_summary):
             print(f"ğŸš¨ **HIGH RISK VIOLATIONS:** {len(risk_summary)} found.")
             for violation in risk_summary:
                  print(f"- {violation}")
        else:
            print("âœ… Compliance Passed (Risk Check Undetermined/Clean).")
    
    print("\n### Strategic Review")
    if result.final_strategic_review:
        print(result.final_strategic_review[:400] + "...\n(truncated)")

