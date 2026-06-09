import os
import logging

from kaggle_secrets import UserSecretsClient
from google.genai import types

# ADK core components
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, AgentTool
from google.adk.tools.tool_context import ToolContext

# Observability plugins
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin

print("✅ Imports loaded.")

# Configure Gemini API key from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ GOOGLE_API_KEY configured from Kaggle Secrets.")
except Exception as e:
    print("❌ ERROR: Please create a Kaggle secret named 'GOOGLE_API_KEY'.")
    raise e

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
print("✅ Logging configured.")



retry_config = types.HttpRetryOptions(
    attempts=5,          # Maximum retry attempts
    exp_base=7,          # Exponential backoff base
    initial_delay=1,     # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504],  # Retry on these errors
)

print("✅ Retry config ready.")



sample_tickets = [
    {
        "id": "TCK-001",
        "text": (
            "Our checkout page keeps timing out with error code 504 when customers try to pay. "
            "This started about 20 minutes ago and is affecting all regions."
        ),
    },
    {
        "id": "TCK-002",
        "text": (
            "I was charged twice for my monthly subscription. "
            "Please refund the duplicate charge and confirm."
        ),
    },
    {
        "id": "TCK-003",
        "text": (
            "I can't log in to my account. It says 'invalid password' even after resetting. "
            "This is blocking me from accessing my reports."
        ),
    },
]

print(f"✅ Loaded {len(sample_tickets)} sample tickets.")



google_search_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="google_search_agent",
    description="Searches the web for information about error codes, outages, and product issues.",
    instruction=(
        "You are a search specialist. "
        "Use the google_search tool to find relevant, up-to-date information "
        "about software issues, error codes, and outages. "
        "Return concise bullet points with links when possible."
    ),
    tools=[google_search],
)

# 🔧 Wrap the sub-agent as a tool for the root agent
google_search_tool = AgentTool(google_search_agent)

print("✅ google_search_agent and AgentTool created.")



TICKET_TRIAGE_SYSTEM_PROMPT = """
You are an AI assistant helping a customer support team triage tickets.

Given a single support ticket text, you MUST:

1) Read and understand the customer's issue.
2) If needed, call the `google_search_agent` tool to look up error codes or incidents.
3) Produce a structured JSON object with the following keys:

- priority: one of ["P1", "P2", "P3"]
  * P1 = critical outage, many users impacted, revenue impact
  * P2 = important but not full outage
  * P3 = minor issue, low impact, question or cosmetic bug

- category: one of ["Billing", "Login", "Performance", "Bug", "Other"]

- needs_escalation: boolean (true/false)

- root_cause_hypothesis: short text explaining what might be causing the issue,
  based on the ticket text and any search results.

- suggested_response: a concise customer-facing message in a professional tone.

Always return ONLY valid JSON. Do not wrap with backticks or extra text.
"""

ticket_triage_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ticket_triage_agent",
    description="Reads support tickets, calls search sub-agent when needed, and returns structured triage output.",
    instruction=TICKET_TRIAGE_SYSTEM_PROMPT,
    tools=[google_search_tool],  # Multi-agent: uses the google_search_agent as a tool
)

print("✅ ticket_triage_agent defined.")



APP_NAME = "customer_support_triage"
USER_ID = "support_agent_demo"

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    agent=ticket_triage_agent,
    session_service=session_service,
    memory_service=memory_service,
    app_name=APP_NAME,
)

print("✅ Runner initialized with sessions + memory services.")



from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.base_agent import BaseAgent

class TicketStatsPlugin(BasePlugin):
    """Tracks number of tickets processed and escalations."""

    def __init__(self):
        super().__init__(name="ticket_stats_plugin")
        self.total_tickets = 0
        self.total_escalations = 0

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ):
        # Only track final responses of the root agent
        if agent.name != "ticket_triage_agent":
            return None

        self.total_tickets += 1

        # Try to parse JSON from the final response
        try:
            from json import loads
            content_list = callback_context.response.content
            # Collect any text parts as a single JSON string
            json_text = "".join(
                part.text
                for msg in content_list
                for part in msg.parts
                if getattr(part, "text", None)
            )
            data = loads(json_text)
            if data.get("needs_escalation") is True:
                self.total_escalations += 1
        except Exception as e:
            logging.warning("TicketStatsPlugin: failed to parse JSON: %s", e)

        logging.info(
            "[TicketStatsPlugin] tickets=%d, escalations=%d",
            self.total_tickets,
            self.total_escalations,
        )
        print(
            f"📊 [TicketStatsPlugin] tickets={self.total_tickets}, escalations={self.total_escalations}"
        )

        return None

ticket_stats_plugin = TicketStatsPlugin()
print("✅ TicketStatsPlugin created.")



runner = Runner(
    agent=ticket_triage_agent,
    session_service=session_service,
    memory_service=memory_service,
    app_name=APP_NAME,
    plugins=[
        LoggingPlugin(),
        ticket_stats_plugin,
    ],
)

print("✅ Runner updated with LoggingPlugin + TicketStatsPlugin.")



import json

async def triage_ticket(ticket_text: str, ticket_id: str):
    print(f"\n=== Triage for Ticket {ticket_id} ===\n")
    
    response = await runner.run_debug(
        ticket_text,
        session_id=ticket_id,
        user_id=USER_ID,
    )

    # Extract JSON text from the response
    try:
        content_list = response.content
        json_text = "".join(
            part.text
            for msg in content_list
            for part in msg.parts
            if getattr(part, "text", None)
        )
        data = json.loads(json_text)
        print("✅ Parsed triage JSON:")
        print(json.dumps(data, indent=2))
        return data
    except Exception as e:
        print("❌ Failed to parse JSON from model response:", e)
        print("Raw response:", response)
        return None

print("✅ triage_ticket helper ready.")



triage_results = []

print("▶ Running triage on a single sample ticket...\n")

# Just use the first sample ticket to keep output small
ticket = sample_tickets[0]

result = await triage_ticket(ticket["text"], ticket["id"])

if result:
    triage_results.append({"id": ticket["id"], "result": result})
    # Compact one-line summary for judges
    print(
        f"\nSummary → Ticket {ticket['id']}: "
        f"priority={result.get('priority')}, "
        f"category={result.get('category')}, "
        f"needs_escalation={result.get('needs_escalation')}"
    )

print("\n✅ Triage demo completed for one sample ticket.")





