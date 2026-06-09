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


import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Created two reference datasets for medical billing validation:
# 
# 1. CHARGE_MASTER: Hospital's internal pricing database
#    - Contains charge codes, descriptions, prices, and CPT codes
#    - Represents what the hospital claims to charge for services
#
# 2. CPT_CODES_REFERENCE: Industry-standard pricing benchmarks
#    - Contains typical price ranges for medical procedures
#    - Used to identify overcharges and billing errors
#
# These datasets simulate real-world hospital billing data

print("\nğŸ“Š Creating manual datasets...")

CHARGE_MASTER = {
    "100761": {"code": "100761", "desc": "TREATMENT ROOM", "dept": "610", "charge": 1933.00, "cpt": "761", "rev_code": ""},
    "700001": {"code": "700001", "desc": "DIRECT REFERRAL TO OBS", "dept": "610", "charge": 1839.00, "cpt": "0G0379", "rev_code": "762"},
    "710060": {"code": "710060", "desc": "IV HYDRAT INIT UP TO 1HR", "dept": "610", "charge": 2165.25, "cpt": "096360", "rev_code": "260"},
    "710061": {"code": "710061", "desc": "IV HYDRAT EA ADD HR", "dept": "610", "charge": 683.50, "cpt": "096361", "rev_code": "260"},
    "710065": {"code": "710065", "desc": "IV INITIAL UP TO 1 HOUR", "dept": "610", "charge": 2165.25, "cpt": "096365", "rev_code": "260"},
    "710066": {"code": "710066", "desc": "IV EACH ADD HOUR", "dept": "610", "charge": 683.50, "cpt": "096366", "rev_code": "260"},
    "710067": {"code": "710067", "desc": "IV ADDL SEQ INF UP TO 1H", "dept": "610", "charge": 2165.25, "cpt": "096367", "rev_code": "260"},
    "820101": {"code": "820101", "desc": "X-RAY CHEST 1 VIEW", "dept": "620", "charge": 425.00, "cpt": "71045", "rev_code": "320"},
    "820102": {"code": "820102", "desc": "X-RAY CHEST 2 VIEWS", "dept": "620", "charge": 550.00, "cpt": "71046", "rev_code": "320"},
    "850200": {"code": "850200", "desc": "CT SCAN HEAD W/O CONTRAST", "dept": "630", "charge": 2850.00, "cpt": "70450", "rev_code": "351"},
    "850201": {"code": "850201", "desc": "CT SCAN HEAD W/ CONTRAST", "dept": "630", "charge": 3200.00, "cpt": "70460", "rev_code": "351"},
    "910050": {"code": "910050", "desc": "COMPLETE BLOOD COUNT", "dept": "640", "charge": 185.00, "cpt": "85025", "rev_code": "300"},
    "910051": {"code": "910051", "desc": "COMPREHENSIVE METABOLIC", "dept": "640", "charge": 225.00, "cpt": "80053", "rev_code": "301"},
}

CPT_CODES_REFERENCE = {
    "71045": {"code": "71045", "desc": "Chest X-Ray, 1 view", "typical_range": (300, 500), "category": "Radiology"},
    "71046": {"code": "71046", "desc": "Chest X-Ray, 2 views", "typical_range": (400, 650), "category": "Radiology"},
    "70450": {"code": "70450", "desc": "CT Head without contrast", "typical_range": (2000, 3000), "category": "CT Scan"},
    "70460": {"code": "70460", "desc": "CT Head with contrast", "typical_range": (2500, 3500), "category": "CT Scan"},
    "85025": {"code": "85025", "desc": "Complete Blood Count (CBC)", "typical_range": (100, 250), "category": "Laboratory"},
    "80053": {"code": "80053", "desc": "Comprehensive Metabolic Panel", "typical_range": (150, 300), "category": "Laboratory"},
    "96365": {"code": "96365", "desc": "IV infusion, initial hour", "typical_range": (1800, 2400), "category": "Infusion"},
    "96366": {"code": "96366", "desc": "IV infusion, each additional hour", "typical_range": (500, 800), "category": "Infusion"},
    "96360": {"code": "96360", "desc": "IV hydration, initial hour", "typical_range": (1800, 2400), "category": "Infusion"},
    "96361": {"code": "96361", "desc": "IV hydration, each additional hour", "typical_range": (500, 800), "category": "Infusion"},
}

print(f"âœ… Charge Master: {len(CHARGE_MASTER)} entries loaded")
print(f"âœ… CPT Reference: {len(CPT_CODES_REFERENCE)} standard codes loaded")


# Create the Charge Master Agent - our first specialized agent
# Purpose: Acts as the hospital's billing database
# Tool: search_charge_master() - looks up charge codes and returns pricing details
# This agent will be exposed via A2A protocol so other agents can query it remotely,
# simulating how a real hospital system might provide billing information to external services

print("\nğŸ�¥ Creating Charge Master Lookup Agent...")

def search_charge_master(charge_code: str) -> str:
    """Search hospital charge master by charge code.
    
    Args:
        charge_code: Hospital internal charge code (e.g., "710060")
    
    Returns:
        Charge master entry details as JSON string
    """
    charge_code = charge_code.strip()
    
    if charge_code in CHARGE_MASTER:
        entry = CHARGE_MASTER[charge_code]
        return json.dumps({
            "found": True,
            "charge_code": entry["code"],
            "description": entry["desc"],
            "department": entry["dept"],
            "hospital_charge": entry["charge"],
            "cpt_code": entry["cpt"],
            "revenue_code": entry["rev_code"]
        })
    else:
        return json.dumps({
            "found": False,
            "error": f"Charge code {charge_code} not found in hospital charge master",
            "suggestion": "Verify the charge code on your bill"
        })

charge_master_agent = LlmAgent(
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    name="charge_master_agent",
    description="Hospital charge master lookup service that provides pricing and billing codes.",
    instruction="""
    You are a hospital charge master database specialist.
    When asked about charge codes, use the search_charge_master tool to look up information.
    Provide accurate charge information including description, price, and associated CPT codes.
    Be precise and professional in your responses.
    """,
    tools=[search_charge_master],
)

print("âœ… Charge Master Agent created")
print("   Model: gemini-2.0 flash")
print("   Tool: search_charge_master()")


# Convert the Charge Master Agent into an A2A-compatible web service
# The agent is now accessible at http://localhost:8001 with a published "agent card" that describes its capabilities, just like a hospital's billing API would work in production
# We use uvicorn to run this agent as a background server, allowing other agents to communicate with it over HTTP using the standardized A2A protocol

print("\nğŸŒ� Converting Charge Master Agent to A2A format...")

charge_master_code = '''
import os
import json
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

CHARGE_MASTER = {
    "100761": {"code": "100761", "desc": "TREATMENT ROOM", "dept": "610", "charge": 1933.00, "cpt": "761", "rev_code": ""},
    "700001": {"code": "700001", "desc": "DIRECT REFERRAL TO OBS", "dept": "610", "charge": 1839.00, "cpt": "0G0379", "rev_code": "762"},
    "710060": {"code": "710060", "desc": "IV HYDRAT INIT UP TO 1HR", "dept": "610", "charge": 2165.25, "cpt": "096360", "rev_code": "260"},
    "710061": {"code": "710061", "desc": "IV HYDRAT EA ADD HR", "dept": "610", "charge": 683.50, "cpt": "096361", "rev_code": "260"},
    "710065": {"code": "710065", "desc": "IV INITIAL UP TO 1 HOUR", "dept": "610", "charge": 2165.25, "cpt": "096365", "rev_code": "260"},
    "710066": {"code": "710066", "desc": "IV EACH ADD HOUR", "dept": "610", "charge": 683.50, "cpt": "096366", "rev_code": "260"},
    "710067": {"code": "710067", "desc": "IV ADDL SEQ INF UP TO 1H", "dept": "610", "charge": 2165.25, "cpt": "096367", "rev_code": "260"},
    "820101": {"code": "820101", "desc": "X-RAY CHEST 1 VIEW", "dept": "620", "charge": 425.00, "cpt": "71045", "rev_code": "320"},
    "820102": {"code": "820102", "desc": "X-RAY CHEST 2 VIEWS", "dept": "620", "charge": 550.00, "cpt": "71046", "rev_code": "320"},
    "850200": {"code": "850200", "desc": "CT SCAN HEAD W/O CONTRAST", "dept": "630", "charge": 2850.00, "cpt": "70450", "rev_code": "351"},
    "850201": {"code": "850201", "desc": "CT SCAN HEAD W/ CONTRAST", "dept": "630", "charge": 3200.00, "cpt": "70460", "rev_code": "351"},
    "910050": {"code": "910050", "desc": "COMPLETE BLOOD COUNT", "dept": "640", "charge": 185.00, "cpt": "85025", "rev_code": "300"},
    "910051": {"code": "910051", "desc": "COMPREHENSIVE METABOLIC", "dept": "640", "charge": 225.00, "cpt": "80053", "rev_code": "301"},
}

def search_charge_master(charge_code: str) -> str:
    charge_code = charge_code.strip()
    if charge_code in CHARGE_MASTER:
        entry = CHARGE_MASTER[charge_code]
        return json.dumps({
            "found": True,
            "charge_code": entry["code"],
            "description": entry["desc"],
            "department": entry["dept"],
            "hospital_charge": entry["charge"],
            "cpt_code": entry["cpt"],
            "revenue_code": entry["rev_code"]
        })
    else:
        return json.dumps({
            "found": False,
            "error": f"Charge code {charge_code} not found",
            "suggestion": "Verify the charge code"
        })

charge_master_agent = LlmAgent(
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    name="charge_master_agent",
    description="Hospital charge master lookup service",
    instruction="You are a hospital charge master database specialist. Use search_charge_master tool to look up codes.",
    tools=[search_charge_master],
)

app = to_a2a(charge_master_agent, port=8001)
'''

with open("/tmp/charge_master_server.py", "w") as f:
    f.write(charge_master_code)

print("ğŸ“� Charge Master agent code saved to /tmp/charge_master_server.py")

# Start uvicorn server
server_process = subprocess.Popen(
    ["uvicorn", "charge_master_server:app", "--host", "localhost", "--port", "8001"],
    cwd="/tmp",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},
)

print("ğŸš€ Starting Charge Master Agent server...")
print("   Waiting for server to be ready...")

# Wait for server
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get("http://localhost:8001/.well-known/agent-card.json", timeout=1)
        if response.status_code == 200:
            print(f"\nâœ… Charge Master Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(1)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet.")

globals()["charge_master_server_process"] = server_process


# Created the CPT Validator Agent - our second specialized agent

# Purpose: Validates medical billing codes and detects overcharges
# Tool: validate_cpt_code() - compares hospital charges against industry standards

# This agent acts as a billing expert, checking if:
# - The CPT code is valid
# - The charged amount is within typical ranges
# - Any overcharges exist (and by how much)
# This is a local sub-agent (not exposed via A2A) that works directly with our main analyzer

print("\nğŸ”� Creating CPT Code Validator Agent...")

def validate_cpt_code(cpt_code: str, charged_amount: float) -> str:
    """Validate CPT code and check if charge is reasonable.
    
    Args:
        cpt_code: CPT procedure code (e.g., "71045")
        charged_amount: Amount charged by hospital
    
    Returns:
        Validation results as JSON string
    """
    cpt_code = cpt_code.strip()
    
    if cpt_code in CPT_CODES_REFERENCE:
        ref = CPT_CODES_REFERENCE[cpt_code]
        min_price, max_price = ref["typical_range"]
        
        is_overcharged = charged_amount > max_price
        is_undercharged = charged_amount < min_price
        variance_pct = ((charged_amount - max_price) / max_price * 100) if is_overcharged else 0
        
        return json.dumps({
            "valid": True,
            "cpt_code": cpt_code,
            "description": ref["desc"],
            "category": ref["category"],
            "typical_range": {"min": min_price, "max": max_price},
            "charged_amount": charged_amount,
            "is_overcharged": is_overcharged,
            "is_undercharged": is_undercharged,
            "variance_percentage": round(variance_pct, 2) if is_overcharged else 0,
            "assessment": "OVERCHARGED" if is_overcharged else ("UNDERCHARGED" if is_undercharged else "REASONABLE")
        })
    else:
        return json.dumps({
            "valid": False,
            "cpt_code": cpt_code,
            "error": f"CPT code {cpt_code} not found in reference database",
            "suggestion": "This may be an invalid or very rare procedure code"
        })

cpt_validator_agent = LlmAgent(
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    name="cpt_validator_agent",
    description="CPT code validator that checks procedure codes and pricing against industry standards.",
    instruction="""
    You are a medical billing expert specializing in CPT code validation.
    Use the validate_cpt_code tool to check if procedure codes are valid and if charges are reasonable.
    Identify overcharges by comparing hospital charges to typical ranges.
    Provide clear assessment of whether charges are legitimate or inflated.
    """,
    tools=[validate_cpt_code],
)

print("âœ… CPT Validator Agent created")
print("   Model: gemini-2.0-flash")
print("   Tool: validate_cpt_code()")



# Create the Dispute Letter Writer Agent - our third specialized agent

# Purpose: Generates professional, legally-sound dispute letters for billing errors
# Tool: generate_dispute_letter() - creates formal letters citing laws and regulations

# When billing errors are found, this agent automatically drafts a complete letter that:
# - Details each error with specific amounts and percentages
# - Cites relevant healthcare transparency laws (No Surprises Act, AB 1045)
# - Provides clear next steps for the patient
# - Is ready to send directly to the hospital billing department

print("\nğŸ“� Creating Dispute Letter Writer Agent...")

def generate_dispute_letter(patient_name: str, account_number: str, errors_json: str) -> str:
    """Generate a professional dispute letter for billing errors.
    
    Args:
        patient_name: Patient's full name
        account_number: Hospital account/bill number
        errors_json: JSON string containing list of billing errors
    
    Returns:
        JSON with dispute letter and summary
    """
    errors = json.loads(errors_json)
    
    error_details = ""
    total_overcharge = 0
    
    for i, error in enumerate(errors, 1):
        error_details += f"\n{i}. {error['description']}\n"
        error_details += f"   - Charge Code: {error['charge_code']}\n"
        error_details += f"   - CPT Code: {error['cpt_code']}\n"
        error_details += f"   - Amount Charged: ${error['charged_amount']:.2f}\n"
        error_details += f"   - Typical Maximum: ${error['typical_max']:.2f}\n"
        error_details += f"   - Overcharge: ${error['overcharge_amount']:.2f} ({error['variance_percentage']}% above maximum)\n"
        total_overcharge += error['overcharge_amount']
    
    letter = f"""
[Your Name]
[Your Address]
[City, State ZIP]
[Email]
[Phone]

[Date]

Hospital Billing Department
[Hospital Name]
[Hospital Address]
[City, State ZIP]

RE: Billing Dispute - Account Number: {account_number}
    Patient Name: {patient_name}

Dear Billing Department,

I am writing to formally dispute charges on my recent hospital bill (Account #{account_number}). After careful review and analysis, I have identified ${total_overcharge:.2f} in incorrect or excessive charges that require immediate adjustment.

DISPUTED CHARGES:
{error_details}

TOTAL DISPUTED AMOUNT: ${total_overcharge:.2f}

SUPPORTING DOCUMENTATION:
- These charges exceed the typical range for these procedures as published in the HCAI Hospital Chargemaster database
- The CPT codes and pricing have been verified against industry standards and Medicare allowable rates
- Similar procedures at comparable facilities in this region are priced significantly lower

REQUEST FOR ACTION:
I respectfully request that you:
1. Review and adjust the above charges to fair market rates within the typical range
2. Provide an itemized explanation for any charges that differ from standard pricing
3. Issue a corrected bill reflecting the adjusted amounts
4. Confirm in writing that no collection activity will occur during this review period

REGULATORY COMPLIANCE:
Under the No Surprises Act and California state billing transparency laws (AB 1045), patients have the right to:
- Receive accurate, fair billing that reflects reasonable and customary rates
- Dispute charges that exceed typical ranges for procedures
- Request detailed explanation of all charges
- Review hospital charge master data

I expect a response within 30 days as required by law. If this matter is not resolved satisfactorily, I am prepared to:
- File a complaint with the California Department of Health Care Access and Information (HCAI)
- Submit a dispute to my insurance company
- Seek assistance from a patient advocacy organization
- Report this matter to the state Attorney General's office

I appreciate your prompt attention to this matter and look forward to resolving this dispute amicably.

Sincerely,

{patient_name}

Enclosures:
- Copy of original bill
- CPT code pricing analysis
- Industry pricing references
- Charge master documentation
"""
    
    return json.dumps({
        "letter": letter,
        "total_overcharge": round(total_overcharge, 2),
        "error_count": len(errors),
        "summary": f"Dispute letter generated for {len(errors)} billing error(s) totaling ${total_overcharge:.2f} in overcharges."
    })

dispute_writer_agent = LlmAgent(
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    name="dispute_writer_agent",
    description="Generates professional dispute letters for incorrect medical bills with legal references.",
    instruction="""
    You are a professional medical billing dispute letter writer.
    Use the generate_dispute_letter tool to create formal, legally sound dispute letters.
    Include all error details, specific amounts, legal references, and patient rights.
    The letters should be professional, firm, and ready to send to the hospital billing department.
    """,
    tools=[generate_dispute_letter],
)

print("âœ… Dispute Letter Writer Agent created")
print("   Model: gemini-2.0-flash")
print("   Tool: generate_dispute_letter()")


# Created the Medical Bill Analyzer - our main orchestrator agent

# This is the primary agent that patients interact with
# It coordinates three specialized sub-agents:
# 1. Charge Master Agent (via A2A) - looks up hospital prices
# 2. CPT Validator Agent (local) - validates codes and detects overcharges  
# 3. Dispute Letter Writer Agent (local) - generates dispute letters when needed

# The agent follows a complete 3-step workflow:
# Step 1: Look up all charges using the remote Charge Master Agent
# Step 2: Validate each charge using the CPT Validator Agent
# Step 3: Generate a professional dispute letter if ANY errors are found

# This demonstrates multi-agent collaboration where specialized agents work together
# to solve a complex problem that no single agent could handle alone

print("\nğŸ�¥ Creating Medical Bill Analyzer Agent...")

# RemoteA2aAgent proxy for Charge Master
remote_charge_master = RemoteA2aAgent(
    name="charge_master_agent",
    description="Remote hospital charge master lookup service",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Charge Master proxy created")

# Main Medical Bill Analyzer with ALL THREE sub-agents
medical_bill_analyzer = LlmAgent(
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    name="medical_bill_analyzer",
    description="AI-powered medical bill analyzer that identifies errors and generates dispute letters.",
    instruction="""
    You are an expert medical billing advocate. Your mission is to clearly and concisely present the findings of a bill analysis in three distinct, easily readable sections, avoiding unnecessary markdown symbols like asterisks or hyphens.
    
    **COMPLETE WORKFLOW - FOLLOW THESE STEPS IN ORDER:**
    
    1. **Lookup Charges**: For each charge code, use charge_master_agent to look up the hospital's pricing and CPT codes.
    
    2. **Validate Pricing**: For each CPT code found, use cpt_validator_agent to:
       - Validate the CPT code is correct
       - Check if the charged amount is within the typical range for that CPT
       - Identify any overcharges with specific amounts and percentages
    
    3. **Generate Dispute Letter**: If ANY overcharges or incorrect codes are found, you MUST use dispute_writer_agent to generate a professional dispute letter by calling generate_dispute_letter with:
       - patient_name (from user input, e.g., "Sarah Johnson")
       - account_number (from user input, e.g., "ACCT-987654")
       - errors_json: A JSON string containing an array of all identified errors.
    
    **CRITICAL**: You MUST complete ALL THREE steps. If you find overcharges, ALWAYS generate the dispute letter. After calling 'generate_dispute_letter', you MUST include the entire content of the generated letter in your final text response.
    
    **OUTPUT FORMAT (MUST BE CLEAN AND CLEAR):**
    
    ============================================================
    1. BILL ANALYSIS SUMMARY
    ============================================================
    [State clearly if the bill is correct or if issues were found.]
    [If incorrect, summarize the total number of errors and the total overcharged amount.]
    
    ============================================================
    2. CHARGE VALIDATION FINDINGS
    ============================================================
    [For each charge, show the Charge Code, CPT Code, Description, Amount Charged, and the finding (Correct/Overcharged/Wrong Code).]
    [Example for Overcharged: Charge Code 710066 (IV EACH ADD HOUR / CPT 96366): Charged $850.00. This is Overcharged by $166.50 (24.36% above the typical maximum of $683.50).]
    
    ============================================================
    3. DISPUTE LETTER (If Applicable)
    ============================================================
    [Insert the COMPLETE letter text here - nothing else after this]
    """,
    sub_agents=[remote_charge_master, cpt_validator_agent, dispute_writer_agent],
)
print("âœ… Medical Bill Analyzer created")
print("   Model: gemini-2.0-flash")
print("   Sub-agents: 3 (Charge Master via A2A + CPT Validator + Dispute Writer)")


# Created the analyze_bill() function to test our multi-agent system

# This function:
# - Sets up a new session for each bill analysis (maintains conversation state)
# - Formats the patient's bill data for the analyzer agent
# - Runs the complete 3-step analysis workflow
# - Streams the results in real-time as the agents work

# The 2-second delay prevents rate limiting from the Gemini API
# The runner manages agent execution and handles all the A2A communication behind the scenes

print("\nğŸ§ª Testing Medical Bill Analyzer System...\n")

async def analyze_bill(bill_items: list[dict], patient_name: str = "John Doe", account_number: str = "ACCT-123456"):
    """Analyze a medical bill with the multi-agent system.
    
    Args:
        bill_items: List of dicts with 'charge_code' and 'amount' keys
        patient_name: Patient's name for the dispute letter
        account_number: Account/bill number
    """
    import asyncio
    
    # Add delay to avoid rate limits
    await asyncio.sleep(2)
    
    # Create session
    session_service = InMemorySessionService()
    app_name = "medical_bill_app"
    user_id = "patient_001"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    
    # Create runner
    runner = Runner(
        agent=medical_bill_analyzer,
        app_name=app_name,
        session_service=session_service
    )
    
    # Format bill for analysis
    bill_text = f"Patient Name: {patient_name}\nAccount Number: {account_number}\n\n"
    bill_text += "I received a hospital bill with these charges:\n"
    for item in bill_items:
        bill_text += f"- Charge Code: {item['charge_code']}, Amount Charged: ${item['amount']:.2f}\n"
    
    # Strengthening the final instruction in the user message
    bill_text += "\n***\n"
    bill_text += f"Please execute the full 3-step analysis for {patient_name} (Account {account_number}): lookup charges, validate pricing, and if any errors are found, **YOU MUST** generate the complete dispute letter and include its *full text* in your final response, following the 3-section output format exactly."
    
    print(f"ğŸ’µ Hospital Bill for {patient_name}")
    print("="*60)
    for item in bill_items:
        print(f"   Charge {item['charge_code']}: ${item['amount']:.2f}")
    print("="*60 + "\n")
    
    print("ğŸ¤– Medical Bill Analyzer Response:")
    
    full_output = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=bill_text)])
    ):
        if event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    # Stream the text output and also capture it for final print
                    print(part.text, end="", flush=True)
                    full_output += part.text
        
    print("\n" + "="*60)
    #return full_output


# Test Case : Analyze a hospital bill containing an overcharge

# Scenario: Patient Sarah Johnson received a bill with 3 charges
# - Charge 820101 ($425) - Chest X-Ray: Should be REASONABLE
# - Charge 710065 ($2165.25) - IV Initial: Should be REASONABLE  
# - Charge 710066 ($850) - IV Additional Hour: OVERCHARGED (should be ~$683.50)

# Expected outcome: Agent should detect the overcharge and generate a dispute letter

print("\nğŸ“‹ TEST CASE 1: Analyzing bill with potential overcharge")
print("-" * 60)

await analyze_bill([
    {"charge_code": "820101", "amount": 425.00},  # Chest X-Ray - reasonable
    {"charge_code": "710065", "amount": 2165.25},  # IV Initial - reasonable
    {"charge_code": "710066", "amount": 850.00},  # IV Add Hour - OVERCHARGED
], patient_name="Sarah Johnson", account_number="ACCT-987654")

#print("\n\nâœ… Testing complete!")

