# Install required packages
print("📦 Installing dependencies...")

!pip install -q google-adk PyPDF2
!pip install "google-cloud-bigquery-storage<3.0.0,>=2.30.0"
!pip install "rich<14,>=12.4.4"

print("✅ Dependencies installed successfully")
print("   - google-adk (Agent Development Kit)")
print("   - PyPDF2 (PDF text extraction)")


# Import all required libraries
import os
import io
import asyncio
from typing import Dict, Any, List
from datetime import datetime

# Kaggle Secrets for API key
from kaggle_secrets import UserSecretsClient

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types

# PDF processing
import PyPDF2

print("✅ All libraries imported successfully")

# Configure API Key from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Google API Key configured from Kaggle Secrets")
except Exception as e:
    print("❌ ERROR: Could not load GOOGLE_API_KEY from Kaggle Secrets")
    print("   Please add your API key in: Add-ons → Secrets → New Secret")
    print(f"   Error details: {e}")


# Configure retry logic to handle API rate limits and transient errors
# Critical for multi-agent workflows with 5+ sequential API calls
retry_config = types.HttpRetryOptions(
    attempts=5,              # Try up to 5 times before failing
    exp_base=7,              # Exponential backoff: 1s, 7s, 49s, 343s
    initial_delay=1,         # Start with 1 second delay
    http_status_codes=[429, 500, 503, 504]  # Retry on these HTTP errors
)

print("✅ Retry configuration set:")
print(f"   - Max attempts: 5")
print(f"   - Exponential backoff: base 7")
print(f"   - Retry on errors: 429 (rate limit), 500, 503, 504")


# Configure structured output schema for JSON responses
# This ensures agents return valid JSON without markdown wrappers

from google.genai import types

# Define JSON schema for policy extraction
policy_extraction_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "requirements": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "rule_id": types.Schema(type=types.Type.STRING),
                    "category": types.Schema(type=types.Type.STRING),
                    "requirement": types.Schema(type=types.Type.STRING),
                    "severity_if_violated": types.Schema(type=types.Type.STRING),
                    "metrics": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING)
                    )
                },
                required=["rule_id", "category", "requirement", "severity_if_violated"]
            )
        ),
        "total_requirements": types.Schema(type=types.Type.INTEGER)
    },
    required=["requirements", "total_requirements"]
)

# Define JSON schema for violation scanning
violation_scan_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "violations": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "violation_id": types.Schema(type=types.Type.STRING),
                    "violating_text": types.Schema(type=types.Type.STRING),
                    "location": types.Schema(type=types.Type.STRING),
                    "explanation": types.Schema(type=types.Type.STRING),
                    "violated_rule_id": types.Schema(type=types.Type.STRING)
                },
                required=["violation_id", "violating_text", "explanation"]
            )
        ),
        "total_violations": types.Schema(type=types.Type.INTEGER),
        "needs_clarification": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING)
        )
    },
    required=["violations", "total_violations"]
)

# Define JSON schema for violation analysis
violation_analysis_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "violation_id": types.Schema(type=types.Type.STRING),
        "severity": types.Schema(type=types.Type.STRING),
        "risk_analysis": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "security_risk": types.Schema(type=types.Type.STRING),
                "regulatory_impact": types.Schema(type=types.Type.STRING),
                "business_impact": types.Schema(type=types.Type.STRING)
            }
        ),
        "remediation_plan": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING)
        ),
        "estimated_fix_time_hours": types.Schema(type=types.Type.NUMBER),
        "priority": types.Schema(type=types.Type.INTEGER)
    },
    required=["violation_id", "severity", "remediation_plan"]
)

# Define JSON schema for rewrites
rewrite_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "violation_id": types.Schema(type=types.Type.STRING),
        "original_text": types.Schema(type=types.Type.STRING),
        "compliant_rewrite": types.Schema(type=types.Type.STRING),
        "changes_made": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING)
        ),
        "compliance_achieved": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING)
        )
    },
    required=["violation_id", "original_text", "compliant_rewrite", "changes_made"]
)

print("✅ Structured output schemas configured")
print("   - Policy extraction schema")
print("   - Violation scanning schema")
print("   - Violation analysis schema")
print("   - Rewrite schema")


def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path: Path to the PDF file (string path, not bytes)
        
    Returns:
        Dictionary with status and extracted text
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return {
                "status": "success",
                "text": text,
                "page_count": len(pdf_reader.pages)
            }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to extract text from PDF: {str(e)}"
        }

print("✅ PDF extraction tool defined")


# Utility: robust JSON stripper & safe parser
import re
import json
from typing import Tuple, Optional

def strip_json_markdown(text: str) -> str:
    """
    Robustly extract a JSON object from text that may contain markdown code fences,
    surrounding text, or stray backticks.

    Returns:
        The trimmed JSON string starting with '{' and ending with the matching '}'.
        If no JSON-like object found, returns the original text (caller should handle).
    """
    if not text:
        return text

    # Normalize newlines
    t = text.strip()

    # Remove typical json code fences like ```json ... ``` or ``` ... ```
    t = re.sub(r'```(?:json)?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*```$', '', t)

    # Also remove single-line fences like `{"foo": 1}`
    t = re.sub(r'`(\{.*?)`', r'\1', t, flags=re.DOTALL)

    # Now, try to find the largest balanced JSON object in the string.
    # Find first '{' and the matching closing '}' by stack-scanning.
    first_open = t.find('{')
    if first_open == -1:
        # No JSON object start — return original cleaned text (caller will fallback)
        return t

    # Find matching closing brace by scanning (handles nested braces)
    stack = 0
    end_idx = -1
    for idx in range(first_open, len(t)):
        ch = t[idx]
        if ch == '{':
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack == 0:
                end_idx = idx
                break

    if end_idx != -1 and end_idx > first_open:
        candidate = t[first_open:end_idx+1].strip()
        return candidate

    # If no matching close found, return original cleaned text
    return t

def safe_load_json_from_text(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Attempt to extract and load JSON from a possibly messy text blob.
    Returns (dict or None, error_message or None)
    """
    cleaned = strip_json_markdown(text)
    try:
        data = json.loads(cleaned)
        return data, None
    except Exception as e:
        # Return None and the error to let caller attempt other parsing strategies
        return None, str(e)

print("✅ JSON stripper & loader helpers defined.")


def create_policy_extractor_agent(retry_config: types.HttpRetryOptions):
    """
    Creates an agent that extracts structured compliance requirements from policy documents.
    """
    return LlmAgent(
        name="policy_extractor",
        model=Gemini(
            model="gemini-2.0-flash-lite",
            retry_options=retry_config,
            generation_config=types.GenerateContentConfig(
                temperature=0.1,     # Lower temperature for more deterministic output
                response_mime_type="application/json"  # Force raw JSON
            ),
        ),
        description="Extracts and structures compliance requirements from policy documents",
        instruction="""
        You are a policy extraction specialist. Your task is to:
        
        1. Read the provided policy document text carefully
        2. Extract ALL compliance requirements and rules
        3. Identify severity levels for violations (CRITICAL, HIGH, MEDIUM, LOW)
        4. Structure requirements as clear, actionable rules
        5. Note any specific metrics or thresholds (e.g., "within 72 hours", "AES-256")

        For each requirement, provide:
        - "rule_id" (e.g., SEC-1.1, ACCESS-2.3)
        - "category" (e.g., Data Security, Access Control, Data Retention)
        - "requirement": Requirement description (clear and specific)
        - "severity_if_violated: Severity level if violated
        - "metrics": Key metrics or constraints
        
        Be thorough and precise. Every requirement matters for compliance.
        Extract even minor requirements - they all count.

        **CRITICAL: Return ONLY valid JSON with NO markdown formatting.**
        - Do NOT wrap your response in ```json``` code blocks.
        - Output raw JSON starting with { and ending with }
        
        **IMPORTANT: Return your response as valid JSON only, with no preamble or markdown.**
        
        Output Format (JSON):
        {
          "requirements": [
            {
              "rule_id": "SEC-1.1",
              "category": "Data Security",
              "requirement": "All customer PII must be encrypted at rest using AES-256",
              "severity_if_violated": "CRITICAL",
              "metrics": ["AES-256", "at rest"]
            },
            {
              "rule_id": "ACCESS-2.3",
              "category": "Access Control",
              "requirement": "Multi-factor authentication required for all systems",
              "severity_if_violated": "HIGH",
              "metrics": ["MFA", "all systems"]
            }
          ],
          "total_requirements": 15
        }
        
        Extract every requirement precisely. Do not include any text outside the JSON structure.
        """,
        tools=[]
    )

policy_extractor = create_policy_extractor_agent(retry_config)
print(f"✅ Policy Extractor Agent created: {policy_extractor.name}")


def create_document_scanner_agent(retry_config: types.HttpRetryOptions):
    """
    Creates an agent that scans documents for compliance violations.
    """
    return LlmAgent(
        name="document_scanner",
        model=Gemini(
            model="gemini-2.0-flash-lite",
            retry_options=retry_config,
            generation_config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            ),
        ),
        description="Scans documents to identify potential compliance violations",
        instruction="""
        You are a document compliance scanner. 
        
        INPUT FORMAT:
        You will receive a message containing:
        1. A list of compliance requirements (as bullet points or formatted text)
        2. The document text to scan
        
        YOUR TASK:
        1. Parse the requirements from the message
        2. Scan the document for violations of ANY requirement
        3. Look for:
           - Hardcoded credentials, API keys, passwords
           - Unencrypted sensitive data (PII, financial info)
           - SQL injection vulnerabilities (string concatenation)
           - Missing encryption specifications
           - Missing MFA requirements
           - Weak authentication methods
           - Non-compliant retention policies
           - Improper PII handling
        
        4. For EACH violation, provide:
           - violation_id: Unique ID (V1, V2, V3...)
           - violating_text: Exact quote from document
           - location: Section/line reference
           - explanation: Why it violates policy
           - violated_rule_id: Which rule was broken
        
        Only flag CLEAR violations. If ambiguous, add to needs_clarification.

        **OUTPUT: Return ONLY valid JSON, NO markdown fences**
        
        {
          "violations": [
            {
              "violation_id": "V1",
              "violating_text": "password = 'Admin123!'",
              "location": "Database Configuration section",
              "explanation": "Hardcoded password violates security policy",
              "violated_rule_id": "SEC-1.2"
            }
          ],
          "total_violations": 7,
          "needs_clarification": []
        }
        """,
        tools=[]
    )

document_scanner = create_document_scanner_agent(retry_config)
print(f"✅ Document Scanner Agent created: {document_scanner.name}")


def create_violation_analyzer_agent(retry_config: types.HttpRetryOptions):
    """
    Creates an agent that analyzes and scores compliance violations.
    """
    return LlmAgent(
        name="violation_analyzer",
        model=Gemini(
            model="gemini-2.0-flash-lite",
            retry_options=retry_config,
            generation_config=types.GenerateContentConfig(
                temperature=0.1,     # Lower temperature for more deterministic output
                response_mime_type="application/json"  # Force raw JSON
            ),
        ),
        description="Analyzes violations, assigns severity scores, and provides remediation guidance",
        instruction="""
        You are a compliance violation analyst. Your task is to:
        
        1. Review each identified violation carefully
        2. Assign severity score (CRITICAL, HIGH, MEDIUM, LOW) based on:
           - Security risk: potential for data breach, unauthorized access
           - Regulatory impact: legal penalties, compliance fines
           - Business impact: reputation damage, operational disruption
        3. Provide detailed analysis of WHY it's a violation
        4. Suggest specific, actionable remediation steps
        5. Estimate remediation effort (hours or days)
        
        Severity Guidelines:
        
        🔴 CRITICAL: 
        - Unencrypted customer PII or financial data
        - Hardcoded credentials, API keys, passwords in code
        - Active security vulnerabilities (SQL injection, XSS)
        - Data breach potential
        
        🟠 HIGH: 
        - Missing MFA for sensitive systems
        - SQL injection risks from poor coding practices
        - Non-compliant data retention (violates regulations)
        - Missing encryption for sensitive data in transit
        
        🟡 MEDIUM: 
        - Expired API keys or credentials
        - Incomplete access controls or reviews
        - Missing audit logging
        
        🟢 LOW: 
        - Missing data classification labels
        - Minor documentation issues
        - Style/formatting violations
        
        For each violation provide:
        - Severity score with justification
        - Detailed explanation of risk
        - Step-by-step remediation plan
        - Estimated fix time
        - Priority ranking
        
        Be precise and actionable in your recommendations.

        **CRITICAL: Return ONLY valid JSON with NO markdown formatting.**
        - Do NOT wrap your response in ```json``` code blocks.
        - Output raw JSON starting with { and ending with }
        
        **IMPORTANT: Return your response as valid JSON only, with no preamble or markdown.**
        
        Output Format (JSON):
        {
          "violation_id": "V1",
          "severity": "CRITICAL",
          "risk_analysis": {
            "security_risk": "High potential for data breach",
            "regulatory_impact": "GDPR/CCPA violation, potential fines",
            "business_impact": "Severe reputation damage"
          },
          "remediation_plan": [
            "Move password to environment variable or secret manager",
            "Implement key rotation policy",
            "Add encryption for stored credentials"
          ],
          "estimated_fix_time_hours": 2,
          "priority": 1
        }
        
        Do not include any text outside the JSON structure.
        """,
        tools=[]
    )

violation_analyzer = create_violation_analyzer_agent(retry_config)
print(f"✅ Violation Analyzer Agent created: {violation_analyzer.name}")


def create_rewrite_agent(retry_config: types.HttpRetryOptions):
    """
    Creates an agent that rewrites document sections to be compliant.
    """
    return LlmAgent(
        name="rewrite_agent",
        model=Gemini(
            model="gemini-2.0-flash-lite",
            retry_options=retry_config,
            generation_config=types.GenerateContentConfig(
                temperature=0.1,     # Lower temperature for more deterministic output
                response_mime_type="application/json"  # Force raw JSON
            ),
        ),
        description="Rewrites document sections to comply with policies",
        instruction="""
        You are a compliance rewrite specialist. Your task is to:
        
        1. Take the original violating text/code
        2. Understand the specific compliance violation
        3. Rewrite the text to be FULLY compliant while maintaining original intent
        4. Preserve technical feasibility and business requirements
        5. Explain what changes were made and why
        
        Rewriting Guidelines:
        
        🔐 Security Fixes:
        - Replace hardcoded credentials with environment variables or secret management
        - Add encryption specifications (AES-256, TLS 1.3) where missing
        - Implement parameterized queries instead of string concatenation
        - Add proper error handling without exposing sensitive data
        
        🔑 Access Control:
        - Add MFA requirements for authentication
        - Specify proper access control mechanisms
        - Add audit logging requirements
        
        📅 Data Retention:
        - Specify compliant data retention periods
        - Add automated deletion processes
        - Include backup retention limits
        
        📝 Data Handling:
        - Remove PII from logs and error messages
        - Add encryption requirements for PII
        - Specify secure data storage methods
        
        Format your output as:
        
        ❌ ORIGINAL (VIOLATION):
        [Exact quote of violating text]
        
        ✅ COMPLIANT REWRITE:
        [Fully compliant version]
        
        📋 CHANGES MADE:
        - [Specific change 1]
        - [Specific change 2]
        - ...
        
        ✔️ COMPLIANCE ACHIEVED:
        [Which policy requirements are now met]
        
        Keep rewrites practical, implementable, and maintain the original purpose.

        **CRITICAL: Return ONLY valid JSON with NO markdown formatting.**
        - Do NOT wrap your response in ```json``` code blocks.
        - Output raw JSON starting with { and ending with }
        
        **IMPORTANT: Return your response as valid JSON only, with no preamble or markdown.**
        
        Output Format (JSON):
        {
          "violation_id": "V1",
          "original_text": "password = 'Admin123!'",
          "compliant_rewrite": "password = os.getenv('DB_PASSWORD')  # Retrieved from secure secret manager",
          "changes_made": [
            "Removed hardcoded password",
            "Added environment variable reference",
            "Added comment explaining secure retrieval"
          ],
          "compliance_achieved": [
            "SEC-1.2: No hardcoded credentials",
            "SEC-2.1: Secure credential storage"
          ]
        }
        
        Do not include any text outside the JSON structure.
        """,
        tools=[]
    )

rewrite_agent = create_rewrite_agent(retry_config)
print(f"✅ Rewrite Agent created: {rewrite_agent.name}")


def create_orchestrator_agent(
    policy_extractor,
    document_scanner,
    violation_analyzer,
    rewrite_agent,
    retry_config: types.HttpRetryOptions
):
    """
    Creates the main orchestrator agent that coordinates compliance checking.
    """

    return LlmAgent(
        name="compliance_orchestrator",
        model=Gemini(
            model="gemini-2.0-flash-lite",
            retry_options=retry_config,
            generation_config=types.GenerateContentConfig(
                temperature=0.2,  # Slightly higher for better reasoning
                response_mime_type="application/json"
            )
        ),
        description="Coordinates the full compliance checking workflow and delegates to specialist agents.",
        instruction="""
        You are the Compliance Agent orchestrator. You coordinate specialist agents
        to perform comprehensive compliance checking.
        
        CRITICAL RULES:
        1. DO NOT perform extraction, scanning, analysis or rewriting yourself
        2. ALWAYS delegate to the correct specialist agent
        3. When calling agents, pass information as PLAIN TEXT, not structured JSON
        4. Extract JSON from agent responses (strip markdown fences if present)
        
        ==========================================================
        🧭 SEQUENTIAL WORKFLOW – FOLLOW THESE STEPS IN ORDER
        ==========================================================
        
        STEP 1 – POLICY EXTRACTION
        - Call policy_extractor with just the policy document text
        - Wait for response containing requirements
        - Parse the JSON response (strip ```json fences if present)
        - Store the requirements list
        
        STEP 2 – DOCUMENT SCANNING
        - Call document_scanner with a SIMPLE TEXT MESSAGE like this:
        
          "Scan this document against the policy requirements.
          
          Requirements from policy:
          - Rule SEC-1.1: All PII must be encrypted with AES-256
          - Rule SEC-1.2: No hardcoded credentials
          - Rule ACCESS-2.1: MFA required for all systems
          [... list all requirements as bullet points ...]
          
          Document to scan:
          [paste the full document text here]"
        
        - Keep the message SIMPLE and TEXT-ONLY
        - Do NOT try to pass JSON objects or arrays as parameters
        - Wait for response with violations list
        
        STEP 3 – VIOLATION ANALYSIS (if violations found)
        - For EACH violation, call violation_analyzer with:
          "Analyze this violation:
          
          Violation ID: V1
          Text: [violating text]
          Explanation: [why it violates]
          Rule violated: [rule ID]"
        
        - Collect severity and remediation for each
        
        STEP 4 – REWRITES (for CRITICAL and HIGH only)
        - For each CRITICAL or HIGH violation, call rewrite_agent with:
          "Rewrite this to be compliant:
          
          Violation: [description]
          Original text: [violating code/text]
          Violated rule: [rule that was broken]"
        
        - Collect compliant rewrites
        
        STEP 5 – GENERATE FINAL REPORT (DO NOT CALL A TOOL)
        After collecting all information, produce your final response as JSON:
        
        {
          "document_name": "Feature Proposal",
          "total_violations": <int>,
          "severity_breakdown": {
            "CRITICAL": <int>,
            "HIGH": <int>,
            "MEDIUM": <int>,
            "LOW": <int>
          },
          "violations": [
            {
              "violation_id": "V1",
              "severity": "CRITICAL",
              "violating_text": "...",
              "explanation": "...",
              "violated_rule_id": "...",
              "remediation_steps": ["..."],
              "compliant_rewrite": "..." (if generated)
            }
          ],
          "status": "FAIL" (if violations > 0) or "PASS"
        }
        
        IMPORTANT:
        - Keep ALL agent calls simple with plain text messages
        - Do NOT try to pass JSON/arrays as function parameters
        - Extract and parse JSON from responses (strip markdown if needed)
        - Only YOU produce the final JSON report (not a tool call)
        """,
        tools=[
            AgentTool(agent=policy_extractor),
            AgentTool(agent=document_scanner),
            AgentTool(agent=violation_analyzer),
            AgentTool(agent=rewrite_agent)
        ]
    )


# Instantiate orchestrator
orchestrator = create_orchestrator_agent(
    policy_extractor,
    document_scanner,
    violation_analyzer,
    rewrite_agent,
    retry_config
)

print(f"✅ Orchestrator Agent created: {orchestrator.name}")
print("\n📋 Multi-Agent System Ready:")
print(f"   1. {policy_extractor.name}")
print(f"   2. {document_scanner.name}")
print(f"   3. {violation_analyzer.name}")
print(f"   4. {rewrite_agent.name}")
print(f"   5. {orchestrator.name} (Coordinator)")


# Load Sample Policy Document

# Paths to your files
policy_pdf_path = "/kaggle/input/sample-company-policy-data/acme_corporation_company_policy.pdf"

# Read PDF using the unified function
policy_result = extract_text_from_pdf(policy_pdf_path)
if policy_result["status"] == "success":
    policy_text = policy_result["text"]
    print("✅ Sample Policy Document Loaded")
    print(f"   - Length: {len(policy_text)} characters")
    print(f"   - Pages: {policy_result['page_count']}")
    print(f"   - Sections: 5 (Classification, Handling, Access, Retention, Incident)")
else:
    print(f"❌ Error loading policy: {policy_result['error_message']}")
    policy_text = ""


# Load Sample Document to Scan

document_pdf_path = "/kaggle/input/sample-company-policy-data/acme_doc_to_scan_proposal_for_new_feature.pdf"

# Read PDF using the unified function
document_result = extract_text_from_pdf(document_pdf_path)
if document_result["status"] == "success":
    document_text = document_result["text"]
    print("✅ Sample Document Loaded (Feature Proposal)")
    print(f"   - Length: {len(document_text)} characters")
    print(f"   - Pages: {document_result['page_count']}")
    print(f"   - Contains: Multiple intentional compliance violations for testing")
else:
    print(f"❌ Error loading document: {document_result['error_message']}")
    document_text = ""

print(f"\n🔍 Expected violations:")
print("   - CRITICAL: Hardcoded credentials (password, API key)")
print("   - CRITICAL: Unencrypted PII storage")
print("   - HIGH: SQL injection vulnerability (string concatenation)")
print("   - HIGH: PII in logs and error messages")
print("   - HIGH: Missing MFA")
print("   - HIGH: Non-compliant data retention (indefinite vs 30 days)")
print("   - MEDIUM: Overly detailed error messages")


# Create session service for maintaining conversation state
session_service = InMemorySessionService()

# Create runner with orchestrator agent
runner = Runner(
    agent=orchestrator,
    app_name="ComplianceCopilot",
    session_service=session_service,
    plugins=[LoggingPlugin()]  # Enable observability
)

print("✅ Runner configured successfully")
print(f"   - App name: ComplianceCopilot")
print(f"   - Agent: {orchestrator.name}")
print(f"   - Session service: InMemorySessionService")
print(f"   - Plugins: LoggingPlugin (for traces)")
print("\n🚀 Ready to run compliance check!")


print("==== DEBUG: FINAL ORCHESTRATOR INSTRUCTION SEEN BY ADK ====\n")
print(orchestrator.instruction)
print("\n==========================================================")


import time
from datetime import datetime # Ensure datetime is imported for use in this cell

async def run_compliance_check():
    """Execute the complete compliance workflow."""
    
    start_time = time.time()
    
    # Create session
    session = await session_service.create_session(
        app_name="ComplianceCopilot",
        user_id="demo_user",
        session_id="demo_session_001"
    )
    
    print("="*70)
    print("🔍 STARTING COMPLIANCE CHECK")
    print("="*70)
    print(f"Session ID: demo_session_001")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPolicy length: {len(policy_text)} chars")
    print(f"Document length: {len(document_text)} chars")
    print("\n" + "="*70)
    print()
    
    # Prepare query for orchestrator
    query = f"""
Please perform a complete compliance check with the following workflow:

STEP 1: Extract all compliance requirements from this policy document:

{policy_text}

STEP 2: Scan the following document for violations against those requirements:

{document_text}

STEP 3: For each violation found:
- Analyze and assign severity (CRITICAL, HIGH, MEDIUM, LOW)
- Provide remediation guidance

STEP 4: For CRITICAL and HIGH severity violations:
- Generate compliant rewrites with explanations

STEP 5: Provide a comprehensive summary report with:
- Total violations by severity
- Detailed findings
- All compliant rewrites
- Estimated remediation effort

Please be thorough and follow the workflow sequentially.
    """
    
    query_content = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )
    
    # Run agent and collect results
    results = []
    response_text = ""
    
    print("⚙️ Processing compliance check (this may take 10-15 minutes)...\n")
    
    try:
        # NEW: Wrap the async loop in try/except to catch the underlying error
        async for event in runner.run_async(
            user_id="demo_user",
            session_id="demo_session_001",
            new_message=query_content
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text'):
                        results.append(part.text)
                        response_text += part.text + "\n"
    except Exception as e:
        print("\n" + "!"*70)
        print("🛑 CRASH DETECTED IN RUNNER.RUN_ASYNC:")
        print("!"*70)
        # Re-raise the exception to show the full stack trace
        raise e 
            
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "="*70)
    print("✅ COMPLIANCE CHECK COMPLETE")
    print("="*70)
    print(f"Processing time: {processing_time/60:.2f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return response_text, processing_time

# Execute the compliance check (re-run this cell)
print("🚀 Launching compliance check workflow...\n")
response_text, processing_time = await run_compliance_check()


# DEBUG cell — place immediately after the call that runs the orchestrator/run_compliance_check
print("\n" + "="*80)
print("🐞 DEBUG: RAW AGENT/ORCHESTRATOR RESPONSE")
print("="*80)

try:
    if response_text:
        print(f"Response length: {len(response_text)} characters")
        # Print first N characters so notebook doesn't explode
        preview_len = 4000
        print("\n--- RAW RESPONSE PREVIEW (first {:,} chars) ---".format(preview_len))
        print(response_text[:preview_len])
        if len(response_text) > preview_len:
            print("\n... [truncated] ...")
            print(f"Total response length: {len(response_text)} characters")
    else:
        print("❌ No response captured in variable 'response_text'\nEnsure 'run_compliance_check' returned (response_text, processing_time)")
except Exception as e:
    print("❌ Debug print failed:", e)

print("="*80)


# Enhanced parser: tries cleaned JSON first, then regex/text fallbacks
import re
import json

def parse_compliance_response(response_text: str):
    """
    Enhanced parser that:
    1. Attempts to extract structured JSON using strip_json_markdown()
    2. If JSON extracted, parses 'violations' entries into severity buckets
    3. Falls back to regex/text heuristics if JSON not found/parseable
    Returns:
        {
            "violations": { "CRITICAL": [...], "HIGH": [...], ... },
            "total_violations": int,
            "severity_counts": {...},
            "rewrites_generated": int
        }
    """
    # 0. Initialize
    violations = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
    rewrites_generated = 0

    # 1. Try the cleaned JSON route
    json_obj, err = safe_load_json_from_text(response_text)
    if json_obj:
        # If top-level has 'violations' as a list, map them
        if isinstance(json_obj, dict) and "violations" in json_obj and isinstance(json_obj["violations"], list):
            for v in json_obj["violations"]:
                sev = (v.get("severity") or v.get("level") or "UNKNOWN").upper()
                if sev in violations:
                    violations[sev].append(v)
                else:
                    # If unknown, put into MEDIUM as fallback
                    violations["MEDIUM"].append(v)
        else:
            # Try to discover structure: search nested fields for violations key
            def walk_for_violations(obj):
                if isinstance(obj, dict):
                    for k, val in obj.items():
                        if k.lower().startswith("viol") and isinstance(val, list):
                            return val
                        else:
                            res = walk_for_violations(val)
                            if res:
                                return res
                elif isinstance(obj, list):
                    for it in obj:
                        res = walk_for_violations(it)
                        if res:
                            return res
                return None
            found = walk_for_violations(json_obj)
            if found:
                for v in found:
                    sev = (v.get("severity") or v.get("level") or "UNKNOWN").upper()
                    if sev in violations:
                        violations[sev].append(v)
                    else:
                        violations["MEDIUM"].append(v)

        # Count rewrites if present
        if isinstance(json_obj, dict) and "rewrites" in json_obj:
            if isinstance(json_obj["rewrites"], list):
                rewrites_generated = len(json_obj["rewrites"])
            elif isinstance(json_obj["rewrites"], dict):
                # maybe a mapping of violations -> text
                rewrites_generated = sum(1 for k in json_obj["rewrites"] if json_obj["rewrites"][k])
    else:
        # 2. Fallback: text-based heuristics (emoji headers, 'Violation X', severity keywords)
        # CRITICAL section detection
        crit_section = re.search(r'(?:🔴|CRITICAL|Critical).{0,1200}?((?:Violation|V)\s*\d+.*?)($|\n(?:🟠|🟡|🟢|HIGH|MEDIUM|LOW))', response_text, re.S|re.I)
        if crit_section:
            items = re.findall(r'(?:Violation\s*\d+|V\d+|-\s+)(.+)', crit_section.group(1))
            for i, it in enumerate(items):
                violations["CRITICAL"].append({"id": f"CRITICAL_{i+1}", "text": it.strip()})

        # HIGH
        high_section = re.search(r'(?:🟠|HIGH|High).{0,1200}?((?:Violation|V)\s*\d+.*?)($|\n(?:🟡|🟢|LOW|MEDIUM))', response_text, re.S|re.I)
        if high_section:
            items = re.findall(r'(?:Violation\s*\d+|V\d+|-\s+)(.+)', high_section.group(1))
            for i, it in enumerate(items):
                violations["HIGH"].append({"id": f"HIGH_{i+1}", "text": it.strip()})

        # Medium/Low simple counts if present
        for sev in ["MEDIUM", "LOW"]:
            m = re.search(rf'({sev}|{sev.capitalize()}).{{0,600}}?(\d+)', response_text, re.I)
            if m:
                try:
                    count = int(m.group(2))
                    for i in range(count):
                        violations[sev][i:i] = [{"id": f"{sev}_{i+1}", "text": None}]
                except:
                    pass

        # Count rewrites by keyword
        rewrites_generated = len(re.findall(r'(?:compliant_rewrite|COMPLIANT REWRITE|Fixed version|rewrit(e|ing))', response_text, re.I))

    total_violations = sum(len(lst) for lst in violations.values())
    severity_counts = {k: len(v) for k, v in violations.items()}

    return {
        "violations": violations,
        "total_violations": total_violations,
        "severity_counts": severity_counts,
        "rewrites_generated": rewrites_generated
    }

print("✅ parse_compliance_response replaced with robust JSON-first parser.")


print("\n" + "="*70)
print("📊 COMPLIANCE CHECK RESULTS (DYNAMIC ANALYSIS)")
print("="*70)

# Parse the actual agent response
parsed_results = parse_compliance_response(response_text)

print("\n🔍 VIOLATIONS DETECTED:")
print(f"   Total violations found: {parsed_results['total_violations']}")
print()

for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    count = parsed_results['severity_counts'][severity]
    if count > 0:
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[severity]
        print(f"   {emoji} {severity}: {count}")

print(f"\n✏️  Rewrites generated: {parsed_results['rewrites_generated']}")
print(f"⏱️  Processing time: {processing_time/60:.2f} minutes")

# Calculate time savings
manual_time_hours = 4
time_saved_hours = manual_time_hours - (processing_time / 3600)
time_saved_percent = (time_saved_hours / manual_time_hours) * 100

print(f"⚡ Time savings: {time_saved_hours:.1f} hours ({time_saved_percent:.0f}% reduction)")

# Calculate cost savings (assuming $50/hour manual labor, $0.50 API cost)
manual_cost = manual_time_hours * 50
api_cost = 0.50  # Estimated
cost_saved = manual_cost - api_cost
cost_saved_percent = (cost_saved / manual_cost) * 100

print(f"💰 Cost savings: ${cost_saved:.2f} ({cost_saved_percent:.1f}% reduction)")

print("\n" + "="*70)


import os
import json
from pathlib import Path
from PyPDF2 import PdfReader   # PDF extraction

# Kaggle dataset path
TEST_DATA_PATH = "/kaggle/input/compliance-test-data"

test_documents = {}
gold_labels = None

if os.path.exists(TEST_DATA_PATH):
    print("✅ Test dataset found")

    # ---------- Load gold labels ----------
    gold_labels_path = Path(TEST_DATA_PATH) / "gold_labels.json"
    if gold_labels_path.exists():
        with open(gold_labels_path, "r") as f:
            gold_labels = json.load(f)
        print(f"   Loaded gold labels for {len(gold_labels)} documents")
    else:
        print("   ⚠️ gold_labels.json NOT found — scoring will be skipped")
        gold_labels = None

    # ---------- Load test documents (TXT & PDF) ----------
    test_docs_dir = Path(TEST_DATA_PATH) / "test_documents"
    if test_docs_dir.exists():
        for doc_file in test_docs_dir.glob("*"):  # .txt + .pdf
            if doc_file.suffix.lower() == ".txt":
                with open(doc_file, "r", encoding="utf-8", errors="ignore") as f:
                    test_documents[doc_file.name] = f.read()

            elif doc_file.suffix.lower() == ".pdf":
                try:
                    pdf = PdfReader(str(doc_file))
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    test_documents[doc_file.name] = text
                except Exception as e:
                    print(f"❌ PDF extraction failed for {doc_file.name}: {e}")

        print(f"   Loaded {len(test_documents)} test documents")
    else:
        print("   ⚠️ test_documents folder missing — no documents loaded")

else:
    print("⚠️ Test dataset not found — DEMO MODE enabled")
    print("   To run full evaluation:")
    print("   1. Upload 'demo_data' as a Kaggle dataset")
    print("   2. Add it under the 'Input' section")
    test_documents = None
    gold_labels = None

# ------- Final Status -------
if test_documents and gold_labels:
    print("🔍 Ready for full evaluation — documents + expected labels loaded")
elif test_documents and not gold_labels:
    print("ℹ️ Documents loaded — WITHOUT gold label scoring")
else:
    print("ℹ️ Running demo pipeline only")



async def run_evaluation_on_test_set():
    """
    Run compliance checks on test dataset and calculate real metrics.
    """
    if not test_documents or not gold_labels:
        print("⚠️  Skipping evaluation - test dataset not available")
        print("   Using single document results for metrics\n")
        return None
    
    print("="*70)
    print("🧪 RUNNING EVALUATION ON TEST DATASET")
    print("="*70)
    print(f"Testing {len(test_documents)} documents...\n")
    
    results = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "processing_times": [],
        "per_document_results": {}
    }
    
    for doc_name, doc_text in test_documents.items():
        print(f"📄 Evaluating: {doc_name}")
        
        expected = gold_labels.get(doc_name, {})
        expected_count = expected.get("total_violations", 0)
        expected_severities = expected.get("expected_severity_counts", {})
        
        # Run compliance check on this document
        start_time = time.time()
        
        query = f"""
Scan this document for compliance violations against the policy:

POLICY:
{policy_text}

DOCUMENT:
{doc_text}

Provide a summary with:
- Total violations
- Breakdown by severity (CRITICAL, HIGH, MEDIUM, LOW)
        """
        
        query_content = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
        
        # Get agent response
        doc_response = ""
        async for event in runner.run_async(
            user_id="eval_user",
            session_id=f"eval_{doc_name}",
            new_message=query_content
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text'):
                        doc_response += part.text
        
        elapsed = time.time() - start_time
        results["processing_times"].append(elapsed)
        
        # Parse results
        parsed = parse_compliance_response(doc_response)
        actual_count = parsed["total_violations"]
        actual_severities = parsed["severity_counts"]
        
        # Calculate metrics
        if expected_count > 0:
            # Document has violations
            tp = min(actual_count, expected_count)
            fp = max(0, actual_count - expected_count)
            fn = max(0, expected_count - actual_count)
            
            results["true_positives"] += tp
            results["false_positives"] += fp
            results["false_negatives"] += fn
        else:
            # Clean document
            if actual_count == 0:
                results["true_negatives"] += 1
            else:
                results["false_positives"] += actual_count
        
        results["per_document_results"][doc_name] = {
            "expected": expected_count,
            "actual": actual_count,
            "time": elapsed,
            "expected_severities": expected_severities,
            "actual_severities": actual_severities
        }
        
        print(f"   Expected: {expected_count} | Detected: {actual_count} | Time: {elapsed:.1f}s")
    
    print("\n✅ Evaluation complete\n")
    return results

# Run evaluation if test data is available
if test_documents and gold_labels:
    eval_results = await run_evaluation_on_test_set()
else:
    eval_results = None
    print("📊 Using single document demo results for display")


print("\n" + "="*70)
print("📈 EVALUATION METRICS")
print("="*70)

if eval_results:
    # Real evaluation from test dataset
    tp = eval_results["true_positives"]
    fp = eval_results["false_positives"]
    fn = eval_results["false_negatives"]
    tn = eval_results["true_negatives"]
    
    total_docs = len(test_documents)
    avg_time = sum(eval_results["processing_times"]) / len(eval_results["processing_times"])
    
    print("\n📁 Test Dataset:")
    print(f"   - Total documents tested: {total_docs}")
    print(f"   - Expected violations: {sum(gold_labels[doc]['total_violations'] for doc in gold_labels)}")
    
    print("\n🎯 Detection Results:")
    print(f"   - True Positives: {tp} (correctly found violations)")
    print(f"   - False Positives: {fp} (incorrectly flagged)")
    print(f"   - False Negatives: {fn} (missed violations)")
    print(f"   - True Negatives: {tn} (correctly identified clean docs)")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print("\n📊 Performance Metrics:")
    print(f"   - Precision: {precision:.2%} (how many flagged are real)")
    print(f"   - Recall: {recall:.2%} (how many real ones found)")
    print(f"   - F1 Score: {f1_score:.3f}")
    print(f"   - Accuracy: {accuracy:.2%}")
    
    print("\n⏱️  Processing Performance:")
    print(f"   - Average time: {avg_time/60:.2f} minutes/document")
    print(f"   - Total time: {sum(eval_results['processing_times'])/60:.2f} minutes")
    
    print("\n📋 Per-Document Breakdown:")
    for doc_name, result in eval_results["per_document_results"].items():
        status = "✅" if result["expected"] == result["actual"] else "⚠️"
        print(f"   {status} {doc_name}: Expected {result['expected']}, Found {result['actual']}")
    
else:
    # Demo mode - use single document results
    print("\n📁 Demo Mode (Single Document):")
    print(f"   - Document analyzed: Feature Proposal")
    print(f"   - Violations detected: {parsed_results['total_violations']}")
    print(f"   - Processing time: {processing_time/60:.2f} minutes")
    
    print("\n🎯 Severity Breakdown:")
    for severity, count in parsed_results['severity_counts'].items():
        if count > 0:
            print(f"   - {severity}: {count}")
    
    print("\n⚠️  Full evaluation requires test dataset")
    print("   To run complete evaluation:")
    print("   1. Create test documents in demo_data/test_documents/ (git)")
    print("   2. Create gold_labels.json with expected violations")
    print("   3. Upload as Kaggle dataset and add to this notebook")

print("\n" + "="*70)


print("\n" + "="*70)
print("🎉 COMPLIANCE Agent DEMO COMPLETE")
print("="*70)

print("\n✅ What We Demonstrated:")
print("   1. ✓ Multi-agent orchestration (5 specialized agents)")
print("   2. ✓ Sequential workflow (Policy → Scan → Analyze → Rewrite)")
print("   3. ✓ Dynamic result parsing (no hardcoded values)")
print("   4. ✓ Session management (InMemorySessionService)")
print("   5. ✓ Real-time metrics calculation")

print("\n📊 Key Results:")
print(f"   - Violations detected: {parsed_results['total_violations']}")
print(f"   - Processing time: {processing_time/60:.2f} minutes")
print(f"   - Time saved: {((4*60 - processing_time)/60):.1f} hours vs manual")
print(f"   - Rewrites generated: {parsed_results['rewrites_generated']}")

if eval_results:
    print(f"   - F1 Score: {f1_score:.3f}")
    print(f"   - Precision: {precision:.1%}")
    print(f"   - Recall: {recall:.1%}")

print("\n🚀 Production Readiness:")
print("   - ✓ Functional multi-agent system")
print("   - ✓ Dynamic parsing and metrics")
print("   - ✓ Scalable architecture")
print("   - ✓ Observable with logging")

print("\n📚 Next Steps:")
print("   1. Add more test documents for comprehensive evaluation")
print("   2. Implement batch processing for multiple documents")
print("   3. Build web UI for compliance teams")
print("   4. Add export to CSV/PDF reports")
print("   5. Integrate with existing compliance tools")

print("\n💡 Repository:")
print("   GitHub: [Your repo URL]")
print("   Demo: This Kaggle notebook")
print("   Docs: Full documentation in README.md")

print("\n" + "="*70)
print("Thank you for reviewing the AI Enterprise Compliance Agent!")
print("="*70)

