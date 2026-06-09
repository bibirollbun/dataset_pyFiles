# Install required packages
!pip install -q google-genai google-adk


# Import required libraries
import os
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
from getpass import getpass

# Google ADK imports
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps import App
from google.adk.tools import AgentTool

print("✅ All imports successful!")


import os
from kaggle_secrets import UserSecretsClient
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    print("✅ Setup and authentication complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# Initialize Gemini model
model = Gemini(model_id="gemini-2.5-flash-lite", api_key=api_key)
print("✅ Gemini 2.5 Flash Lite model initialized")


from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

class SimpleLoggingPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="simple_logging")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ):
        print("🧠 [MODEL] Calling LLM...")

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict, tool_context: ToolContext
    ):
        print(f"🔧 [TOOL] Calling tool '{tool.name}' with args: {tool_args}")

    async def on_tool_error_callback(
        self, *, tool: BaseTool, tool_args: dict, tool_context: ToolContext, error: Exception
    ):
        print(f"❌ [ERROR] Tool '{tool.name}' raised error: {error}")

class ScamMetricsPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="scam_metrics")
        self.stats = {
            "messages_analyzed": 0,
            "tools_used": {}
        }

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict, tool_context: ToolContext
    ):
        name = tool.name
        self.stats["tools_used"][name] = self.stats["tools_used"].get(name, 0) + 1

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response
    ):
        # This is a simplification: counting each response as a “message analyzed”
        self.stats["messages_analyzed"] += 1

    def report(self):
        print("\n📊 --- LIVE SESSION METRICS ---")
        print(f"Messages Analyzed: {self.stats['messages_analyzed']}")
        print("Tool Usage Breakdown:")
        for tool, count in self.stats["tools_used"].items():
            print(f"  - {tool}: {count}")

print("✅ Custom ADK Plugins defined")



# Tool 1: URL Extractor
def extract_urls(message: str) -> Dict[str, Any]:
    """Extract URLs from message text."""
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    urls = re.findall(url_pattern, message)
    return {
        "urls": urls,
        "count": len(urls),
        "has_urls": len(urls) > 0
    }

url_extractor_tool = FunctionTool(func=extract_urls)
print("✅ URL Extractor Tool created")


# Tool 2: Domain Analyzer
def analyze_domain(url: str) -> Dict[str, Any]:
    """Analyze a URL's domain for suspicious patterns."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Suspicious indicators
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work']
        has_suspicious_tld = any(domain.endswith(tld) for tld in suspicious_tlds)

        is_ip = re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) is not None
        subdomain_count = domain.count('.')

        brand_keywords = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'bank', 'secure', 'verify']
        has_brand_keyword = any(keyword in domain for keyword in brand_keywords)

        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
        is_shortener = any(shortener in domain for shortener in shorteners)

        # Calculate suspicion score
        suspicion_score = 0
        if has_suspicious_tld: suspicion_score += 30
        if is_ip: suspicion_score += 40
        if subdomain_count > 3: suspicion_score += 20
        if has_brand_keyword and not any(trusted in domain for trusted in ['.com', '.org', '.gov']):
            suspicion_score += 25
        if is_shortener: suspicion_score += 15

        return {
            "domain": domain,
            "suspicion_score": min(suspicion_score, 100),
            "flags": {
                "suspicious_tld": has_suspicious_tld,
                "is_ip_address": is_ip,
                "excessive_subdomains": subdomain_count > 3,
                "brand_impersonation_risk": has_brand_keyword,
                "url_shortener": is_shortener
            }
        }
    except Exception as e:
        return {"domain": "invalid", "suspicion_score": 50, "error": str(e)}

domain_analyzer_tool = FunctionTool(func=analyze_domain)
print("✅ Domain Analyzer Tool created")


# Tool 3: Risk Scorer
def calculate_risk_score(message: str) -> Dict[str, Any]:
    """Calculate risk score based on message content heuristics."""
    message_lower = message.lower()

    # Scam indicators
    urgency_keywords = ['urgent', 'immediately', 'act now', 'limited time', 'expires today',
                        'hurry', 'quick', 'asap', 'time sensitive', 'deadline']
    money_keywords = ['wire transfer', 'gift card', 'bitcoin', 'cryptocurrency', 'payment',
                     'refund', 'prize', 'lottery', 'inheritance', 'tax refund', 'irs',
                     'bank account', 'credit card', 'ssn', 'social security']
    impersonation_keywords = ['verify account', 'confirm identity', 'suspended', 'locked',
                             'unusual activity', 'security alert', 'click here',
                             'update information', 'validate', 'authenticate', 'reactivate']
    threat_keywords = ['legal action', 'arrest', 'lawsuit', 'police', 'fbi', 'irs audit',
                      'warrant', 'investigation', 'penalty', 'fine']

    # Count indicators
    urgency_count = sum(1 for kw in urgency_keywords if kw in message_lower)
    money_count = sum(1 for kw in money_keywords if kw in message_lower)
    impersonation_count = sum(1 for kw in impersonation_keywords if kw in message_lower)
    threat_count = sum(1 for kw in threat_keywords if kw in message_lower)

    has_excessive_caps = sum(1 for c in message if c.isupper()) > len(message) * 0.3
    has_excessive_exclamation = message.count('!') > 3

    # Calculate risk score
    risk_score = (urgency_count * 10 + money_count * 15 +
                 impersonation_count * 12 + threat_count * 20)
    if has_excessive_caps: risk_score += 10
    if has_excessive_exclamation: risk_score += 8
    risk_score = min(risk_score, 100)

    # Build pattern list
    detected_patterns = []
    if urgency_count > 0:
        detected_patterns.append(f"Urgency tactics ({urgency_count} indicators)")
    if money_count > 0:
        detected_patterns.append(f"Money/financial requests ({money_count} indicators)")
    if impersonation_count > 0:
        detected_patterns.append(f"Account verification requests ({impersonation_count} indicators)")
    if threat_count > 0:
        detected_patterns.append(f"Threats/intimidation ({threat_count} indicators)")
    if has_excessive_caps:
        detected_patterns.append("Excessive capitalization")
    if has_excessive_exclamation:
        detected_patterns.append("Excessive exclamation marks")

    return {
        "risk_score": risk_score,
        "detected_patterns": detected_patterns,
        "pattern_counts": {
            "urgency": urgency_count,
            "money": money_count,
            "impersonation": impersonation_count,
            "threats": threat_count
        }
    }

risk_scorer_tool = FunctionTool(func=calculate_risk_score)
print("✅ Risk Scorer Tool created")


# Agent 1: Classifier Agent
classifier_agent = LlmAgent(
    model=model,
    name="classifier_agent",
    instruction="""
You are a scam classification expert. Your job is to analyze messages and URLs to determine if they are:
- Safe: Legitimate communication with no red flags
- Suspicious: Contains some concerning elements but not definitively malicious
- Likely Scam: Clear indicators of phishing, fraud, or scam attempts
- Misinformation: Contains false or misleading claims

Use the available tools to:
1. Extract URLs from the message
2. Analyze domains for suspicious patterns
3. Calculate risk score based on message content

Based on the tool results, provide:
- A risk label (Safe/Suspicious/Likely Scam/Misinformation)
- An overall risk score (0-100)
- Key findings from your analysis

Be thorough but concise in your analysis.
""",
    tools=[url_extractor_tool, domain_analyzer_tool, risk_scorer_tool]
)

print("✅ Classifier Agent created")


# Agent 2: Explanation Agent
explanation_agent = LlmAgent(
    model=model,
    name="explanation_agent",
    instruction="""
You are an expert at explaining security risks in simple, user-friendly language.

Given the classification results and detected patterns, create a clear explanation that:
1. Explains WHY the message is risky (or safe) in plain English
2. Highlights the specific red flags found
3. Avoids technical jargon - explain like you're talking to a non-technical friend
4. Keeps it brief (2-4 sentences)

Focus on helping the user understand the risk without overwhelming them with details.
""",
    tools=[]
)

print("✅ Explanation Agent created")


# Agent 3: Action Planner Agent
action_planner_agent = LlmAgent(
    model=model,
    name="action_planner_agent",
    instruction="""
You are an expert at recommending security actions based on risk levels.

Given the risk label and context, recommend specific next steps:

For Safe messages:
- Confirm it's okay to proceed normally
- Mention any minor precautions if needed

For Suspicious messages:
- Recommend verification through official channels
- Suggest not clicking links until verified
- Provide alternative ways to check legitimacy

For Likely Scam messages:
- Strongly advise not to respond or click links
- Recommend reporting to IT/security team
- Suggest deleting the message

For Misinformation:
- Recommend fact-checking with trusted sources
- Suggest not sharing the information
- Provide guidance on verifying claims

Keep recommendations actionable and specific (3-5 bullet points).
""",
    tools=[]
)

print("✅ Action Planner Agent created")


# Manager Agent with Memory Capabilities
manager_agent = LlmAgent(
    model=model,
    name="scam_guardian",
    instruction="""
You are the Scam Guardian Manager. Your job is to orchestrate the analysis workflow:

1. First, use the classifier_agent to analyze the message and determine the risk level
2. Then, use the explanation_agent to create a user-friendly explanation of the findings
3. Finally, use the action_planner_agent to recommend specific next steps

Combine all results into a comprehensive response with:
- Risk Label: [Safe / Suspicious / Likely Scam / Misinformation]
- Risk Score: [0-100]
- Explanation: [2-4 sentences explaining WHY. Mention specific red flags found.]
- Recommended Actions: [3-5 bullet points with specific next steps]

**For FOLLOW-UP questions:**
- Provide context-aware advice based on conversation history
- Reference specific past analyses when relevant

Always follow this workflow in order. Be thorough but efficient. Be clear, helpful, and provide actionable insights!
""",
    tools=[AgentTool(agent=classifier_agent), AgentTool(agent=explanation_agent), AgentTool(agent=action_planner_agent)],
)

print("✅ Scam Guardian Agent created")


# Initialize Session and Memory Services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Initialize Plugins
logging_plugin = SimpleLoggingPlugin()
metrics_plugin = ScamMetricsPlugin()

print("✅ Session service initialized")
print("✅ Memory service initialized")

APP_NAME = "scam_guardian"
USER_ID = "demo_user"

# Create App with Plugins
app = App(
    name=APP_NAME,
    root_agent=manager_agent,
    plugins=[logging_plugin, metrics_plugin]
)

runner = Runner(
    app=app,
    session_service=session_service,
    memory_service=memory_service
)

print("✅ Runner initialized with session + memory services")


async def consolidate_and_store_memory(session_id: str):
    """
    Consolidate session into memory (saves tokens!)

    Instead of storing full conversation, we extract key information:
    - Scam type detected
    - Risk score
    - Key red flags
    - User questions asked
    """
    try:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )

        # Add session to memory
        await memory_service.add_session_to_memory(session)

        print(f"💾 Session {session_id} consolidated and stored in memory")
        return True
    except Exception as e:
        print(f"⚠️ Could not store memory: {e}")
        return False

print("✅ Memory consolidation helper ready")


import uuid
import asyncio

class MemoryEnabledConversation:
    """Manages conversation with proper ADK Memory"""

    def __init__(self):
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.message_count = 0
        self.created_at = datetime.now()
        print(f"\n🆕 New conversation session: {self.session_id}")
        print(f"📅 Started at: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    async def ensure_session(self):
        """Create session if it doesn't exist"""
        try:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=self.session_id
            )
        except:
            try:
                await session_service.get_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=self.session_id
                )
            except:
                pass

    async def analyze(self, message: str, show_details: bool = True):
        """Analyze a message with memory"""
        self.message_count += 1

        if show_details:
            print("\n" + "="*80)
            print(f"🛡️ SCAM GUARDIAN ANALYSIS (Message #{self.message_count})")
            print("="*80)
            print(f"\n💬 Input: {message}\n")
            print("-"*80)

        # Create message
        msg = types.Content(role="user", parts=[types.Part(text=message)])

        # Ensure session exists (AWAIT DIRECTLY)
        await self.ensure_session()

        # Run analysis with memory
        events = []
        for event in runner.run(
            new_message=msg,
            user_id=USER_ID,
            session_id=self.session_id
        ):
            events.append(event)

        # Extract and display results
        response_text = ""
        for event in reversed(events):
            if hasattr(event, 'content'):
                content = event.content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text
                            break
                if response_text:
                    break

        if show_details:
            print("\n🤖 Analysis Result:")
            print("-"*80)
            print(response_text if response_text else "No response generated")
            print("\n" + "="*80)

        return response_text, events

    async def save_to_memory(self):
        """Consolidate and save this conversation to long-term memory"""
        print(f"\n💾 Consolidating session to memory...")

        success = await consolidate_and_store_memory(self.session_id)

        if success:
            print("✅ Conversation saved to long-term memory!")

        return success

    def summary(self):
        """Show conversation summary"""
        duration = datetime.now() - self.created_at
        print(f"\n📊 Conversation Summary:")
        print(f"   Session ID: {self.session_id}")
        print(f"   Messages: {self.message_count}")
        print(f"   Duration: {duration.seconds}s")

print("✅ Memory-enabled conversational interface ready")


# Create a new conversation
conv1 = MemoryEnabledConversation()

# Test Case 1: Phishing with suspicious URL
phishing_message = """URGENT: Your account has been suspended!
Click here immediately to verify your identity: http://secure-bank-verify.xyz/account
Failure to act within 24 hours will result in permanent account closure."""

# response1, events1 = conv1.analyze(phishing_message)
response1, events1 = await conv1.analyze(phishing_message)


# Save to memory for future reference
await conv1.save_to_memory()
conv1.summary()

metrics_plugin.report()


conv2 = MemoryEnabledConversation()

# Test Case 2: Gift card scam
gift_card_scam = """Hi! This is your CEO. I'm in an urgent meeting and need you to purchase
$500 in iTunes gift cards immediately. Please send me the codes ASAP.
Don't call me, just text the codes to this number: 555-0123."""

response2, events2 = await conv2.analyze(phishing_message)


await conv2.save_to_memory()
conv2.summary()

metrics_plugin.report()


conv3 = MemoryEnabledConversation()

# Test Case 3: Legitimate message
safe_message = """Hi team,
Just a reminder that our quarterly review meeting is scheduled for next Tuesday at 2 PM.
Please review the attached documents before the meeting.
Best regards,
Sarah"""

response3, events3 = await conv3.analyze(phishing_message)


await conv3.save_to_memory()
conv3.summary()

metrics_plugin.report()


conv4 = MemoryEnabledConversation()

# Test Case 4: Misinformation
misinfo_message = """BREAKING: Scientists confirm that drinking bleach cures all diseases!
The government doesn't want you to know this! Share before this gets deleted!
Click here for more info: http://real-health-news.tk/cure"""

response4, events4 = await conv4.analyze(phishing_message)


await conv4.save_to_memory()
conv4.summary()

metrics_plugin.report()


# Continue conversation 1 to demonstrate memory
followup_response, _ = await conv1.analyze(
    "What should I do if I already clicked the link?",
    show_details=True
)


await conv1.save_to_memory()
conv1.summary()

metrics_plugin.report()


# Interactive test - modify the message below
custom_conv = MemoryEnabledConversation()

your_message = """Enter your test message here..."""

# Uncomment to run:
# custom_response, _ = await custom_conv.analyze(your_message)
# await custom_conv.save_to_memory()
# custom_conv.summary()

# metrics_plugin.report()


# Evaluation Test Suite
evaluation_cases = [
    {
        "id": 1,
        "message": "URGENT: Your PayPal account will be closed! Verify now: http://paypal-secure.xyz/verify",
        "expected_label": "Likely Scam",
        "expected_min_score": 60,
        "category": "Phishing"
    },
    {
        "id": 2,
        "message": "Hi, your package is ready for pickup at the post office. Tracking: 1234567890",
        "expected_label": "Safe",
        "expected_min_score": 0,
        "category": "Legitimate"
    },
    {
        "id": 3,
        "message": "You've won $1,000,000! Send us your bank details and SSN to claim your prize!",
        "expected_label": "Likely Scam",
        "expected_min_score": 70,
        "category": "Prize Scam"
    },
    {
        "id": 4,
        "message": "IRS FINAL NOTICE: You owe $5,000 in back taxes. Pay immediately or face arrest!",
        "expected_label": "Likely Scam",
        "expected_min_score": 75,
        "category": "Tax Scam"
    },
    {
        "id": 5,
        "message": "Meeting rescheduled to 3 PM tomorrow. See you then!",
        "expected_label": "Safe",
        "expected_min_score": 0,
        "category": "Legitimate"
    },
    {
        "id": 6,
        "message": "COVID vaccine contains microchips! Don't get vaccinated! Share this truth!",
        "expected_label": "Misinformation",
        "expected_min_score": 50,
        "category": "Misinformation"
    }
]

print(f"📋 Loaded {len(evaluation_cases)} test cases")


# Run evaluation
import re

def extract_risk_info(response_text):
    """Extract risk label and score from response"""
    label_match = re.search(r'\*\*Risk Label:\*\*\s*\[?([^\]\n]+)', response_text)
    score_match = re.search(r'\*\*Risk Score:\*\*\s*\[?(\d+)', response_text)

    label = label_match.group(1).strip() if label_match else "Unknown"
    score = int(score_match.group(1)) if score_match else 0

    return label, score

print("🧪 Running evaluation suite...\n")

results = []
correct_predictions = 0

for case in evaluation_cases:
    print(f"\n{'='*80}")
    print(f"Test Case #{case['id']}: {case['category']}")
    print(f"{'='*80}")

    eval_conv = MemoryEnabledConversation()
    response, _ = await eval_conv.analyze(case['message'], show_details=False)

    # Extract results
    actual_label, actual_score = extract_risk_info(response)

    # Check if prediction is correct
    is_correct = case['expected_label'].lower() in actual_label.lower()
    if is_correct:
        correct_predictions += 1

    result = {
        "id": case['id'],
        "category": case['category'],
        "expected_label": case['expected_label'],
        "actual_label": actual_label,
        "actual_score": actual_score,
        "correct": is_correct
    }
    results.append(result)

    # Display result
    status = "✅ PASS" if is_correct else "❌ FAIL"
    print(f"\n{status}")
    print(f"Expected: {case['expected_label']}")
    print(f"Actual: {actual_label} (Score: {actual_score})")
    print(f"\nMessage: {case['message'][:100]}...")

print(f"\n\n{'='*80}")
print("📊 EVALUATION SUMMARY")
print(f"{'='*80}")
print(f"Total Test Cases: {len(evaluation_cases)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {(correct_predictions / len(evaluation_cases) * 100):.1f}%")
print(f"{'='*80}")


# Detailed results table
import pandas as pd

results_df = pd.DataFrame(results)
print("\n📋 Detailed Results:")
print(results_df.to_string(index=False))

