# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



from typing import Any, Dict, List
import textwrap

# Core ADK agent types and model
from google.adk.agents import Agent, LlmAgent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini

# Execution runtimes
from google.adk.runners import InMemoryRunner, Runner

# Sessions & state management
from google.adk.sessions import InMemorySessionService

# Observability plugin
from google.adk.plugins.logging_plugin import LoggingPlugin

# Tools
from google.adk.tools import google_search

# A2A helpers
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Gemini base types (retry options etc.)
from google.genai import types

print("âœ… ADK components imported successfully.")



MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def build_model() -> Gemini:
    """Create a Gemini model instance with a shared retry configuration.
    """
    return Gemini(model=MODEL_NAME, retry_options=retry_config)

print(f"âœ… Model helper ready with: {MODEL_NAME}")



def detect_gdpr_issues(text: str) -> Dict[str, Any]:
    """Detect potential GDPR-style compliance issues in free text.

    This function is intentionally simple and rule-based. It does not try
    to implement full legal logic. Instead, it looks for the presence or
    absence of a few key concepts that often appear in privacy and security
    policies:

    - Lawful basis for processing (e.g. consent, legitimate interest)
    - Data retention or storage duration
    - Data subject rights (e.g. deletion / erasure)
    - Access control to personal data
    - Encryption or similar security safeguards
    """
    if not text:
        return {
            "status": "error",
            "error_message": "Empty text provided to detect_gdpr_issues",
        }

    lower = text.lower()
    issues: List[Dict[str, Any]] = []

    def add_issue(category: str, severity: str, reason: str) -> None:
        """Append a new issue in a consistent format."""
        issues.append(
            {
                "category": category,
                "severity": severity,
                "reason": reason,
            }
        )

    # Very simple pattern checks â€” these are placeholders for real policy rules.
    if "consent" not in lower and "legitimate interest" not in lower:
        add_issue(
            "Lawful Basis",
            "High",
            "No mention of consent or legitimate interest as lawful basis.",
        )

    if "retain" not in lower and "retention" not in lower and "store" in lower:
        add_issue(
            "Data Retention",
            "Medium",
            "Data storage is mentioned but no clear retention period is described.",
        )

    if "delete" not in lower and "erasure" not in lower:
        add_issue(
            "Data Subject Rights",
            "Medium",
            "Right to deletion / erasure is not clearly stated.",
        )

    if "access control" not in lower and "role-based" not in lower and "rbac" not in lower:
        add_issue(
            "Access Control",
            "Medium",
            "No explicit mention of restricted / role-based access to personal data.",
        )

    if "encrypt" not in lower and "encryption" not in lower:
        add_issue(
            "Security",
            "Low",
            "Encryption is not mentioned for stored or transmitted data.",
        )

    if not issues:
        add_issue(
            "No obvious issues",
            "Low",
            "Heuristic scanner did not find common GDPR red flags.",
        )

    return {"status": "success", "issues": issues}



def risk_summary(issues: List[Dict[str, Any]]) -> Dict[str, str]:
    """Summarize an overall risk level from individual issues.

    Simple aggregation rule:
    - If any issue is High â†’ overall risk = High
    - Else if any issue is Medium â†’ overall risk = Medium
    - Else â†’ overall risk = Low
    """
    if not issues:
        return {
            "overall_risk": "Low",
            "rationale": "No issues were provided to risk_summary.",
        }

    severities = {issue.get("severity", "Low") for issue in issues}
    if "High" in severities:
        overall = "High"
    elif "Medium" in severities:
        overall = "Medium"
    else:
        overall = "Low"

    return {
        "overall_risk": overall,
        "rationale": f"Aggregated from severities: {sorted(severities)}",
    }



def build_structured_report(policy_name: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a structured JSON-like report for downstream systems.

    This function behaves like the â€œtoolâ€� side of the report agent:
    it takes issues from `detect_gdpr_issues` (or similar) and wraps
    them in a predictable object that other systems (dashboards,
    ticketing, etc.) can consume.
    """
    summary = risk_summary(issues)

    return {
        "policy_name": policy_name,
        "overall_risk": summary["overall_risk"],
        "risk_rationale": summary["rationale"],
        "issue_count": len(issues),
        "issues": issues,
    }

print("âœ… Compliance tools defined and documented.")



# 4.1: Compliance Analyzer Agent

compliance_analyzer_agent = LlmAgent(
    name="ComplianceAnalyzer",
    model=build_model(),
    instruction=textwrap.dedent(
        """                    You are a GDPR-inspired compliance assistant.

        Your job is to:
        1. Carefully read the policy text provided by the user.
        2. When appropriate, call the `detect_gdpr_issues` tool to analyze the text.
        3. Return a concise natural-language summary of the issues found.
        4. Explicitly mention when no obvious issues were detected.

        Always be conservative and transparent: this is not legal advice.
        """
    ),
    tools=[detect_gdpr_issues],
)

print("âœ… ComplianceAnalyzer agent created.")



# 4.2: Policy Context Agent

policy_context_agent = LlmAgent(
    name="PolicyContextAgent",
    model=build_model(),
    instruction=textwrap.dedent(
        """                    You are a policy context researcher.

        - When asked, you use the `google_search` tool to look up short,
          high-level explanations of relevant GDPR concepts.
        - You then summarize them in your own words, keeping the content brief and practical.

        Do not provide legal advice; instead, provide concise background and emphasize
        that formal legal review is required for decisions.
        """
    ),
    tools=[google_search],
)

print("âœ… PolicyContextAgent created.")



# 4.3: Report Agent

report_agent = LlmAgent(
    name="ReportAgent",
    model=build_model(),
    instruction=textwrap.dedent(
        """                    You are a reporting assistant that takes detected issues and produces:

        1. A short executive summary for non-technical stakeholders.
        2. A bullet list of key risks.
        3. Suggestions for next steps (in plain language).

        When you use the `build_structured_report` tool, keep the JSON object unchanged.
        After the JSON, write a clear narrative explanation of the main risks and what
        the organization might consider doing next.
        """
    ),
    tools=[build_structured_report],
)

print("âœ… ReportAgent created.")



# 4.4: Refinement loop and audit pipeline

refinement_loop = LoopAgent(
    name="ReportRefinementLoop",
    sub_agents=[report_agent],
    max_iterations=2,
)

audit_pipeline = SequentialAgent(
    name="ComplianceAuditPipeline",
    sub_agents=[compliance_analyzer_agent, policy_context_agent, refinement_loop],
)

print("âœ… Multi-agent audit pipeline created.")



APP_NAME = "enterprise_compliance_app"
USER_ID = "demo_user"

session_service = InMemorySessionService()

runner = Runner(
    agent=audit_pipeline,
    app_name=APP_NAME,
    session_service=session_service,
)

print("âœ… Runner with InMemorySessionService initialized.")



import asyncio
from google.genai import types as genai_types

async def run_compliance_session(
    runner_instance: Runner,
    user_queries: List[str] | str,
    session_name: str = "default",
) -> None:
    """Simulate a full compliance review session.

    These helper functions:
    - Create or retrieve a session
    - Send one or more user messages
    - Stream back the agent responses
    """
    print(f"\n### Session: {session_name}")

    try:
        session = await session_service.create_session(
            app_name=runner_instance.app_name,
            user_id=USER_ID,
            session_id=session_name,
        )
    except Exception:
        session = await session_service.get_session(
            app_name=runner_instance.app_name,
            user_id=USER_ID,
            session_id=session_name,
        )

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        print(f"\nUser > {query}\n")
        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=query)],
        )

        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                txt = event.content.parts[0].text
                if txt and txt != "None":
                    print(f"{MODEL_NAME} > {txt}\n")

print("âœ… Session helper function defined.")



debug_runner = InMemoryRunner(
    agent=audit_pipeline,
    plugins=[LoggingPlugin()],
)

print("âœ… Debug runner with LoggingPlugin ready.")



sample_policy = """            We collect user data to improve our services. Data is stored indefinitely.
We may share data with partners. Users can contact us for questions.
"""

print("ğŸš€ Running debug audit with logging...\n")

response = await debug_runner.run_debug(
    f"""
    You are the ComplianceAuditPipeline.
    Please audit the following policy text and highlight potential GDPR-style issues.

    POLICY NAME: Demo Policy
    POLICY TEXT:
    {sample_policy}
    """
)

print("\nâœ… Debug run completed.")



test_policies: Dict[str, str] = {
    "No Personal Data": """                We only process anonymized, aggregated statistics. No individual user data is stored.
    """
    ,
    "Missing Retention": """                We store user account information and logs to provide our services.
    """
    ,
    "High Risk: Broad Sharing": """                We collect user data, including identifiers and behavior, and may share it with partners
    and third parties for analytics and advertising without explicit consent.
    Data is stored indefinitely.
    """
}

async def evaluate_policies() -> None:
    """Run the compliance pipeline on a small suite of example policies."""
    for name, text in test_policies.items():
        print(f"\n=== Evaluating: {name} ===")
        await run_compliance_session(
            runner,
            [
                f"""
                Please audit the following policy.
                POLICY NAME: {name}
                POLICY TEXT:
                {text}
                """
            ],
            session_name=name.replace(" ", "_"),
        )

print("âœ… Test policies ready. Call `await evaluate_policies()` to run the evaluation loop.")



def create_a2a_agent():
    """Wrap the audit pipeline as an A2A-compatible agent service (conceptual).

    In a real deployment this function would live inside a small web service which:
    - Exposes the agent card at the well-known path
    - Accepts A2A requests from other clients
    """
    a2a_agent = to_a2a(audit_pipeline)
    return a2a_agent

print("âœ… A2A wrapper helper defined (conceptual).")



example_remote_client = RemoteA2aAgent(
    name="RemoteComplianceAuditAgent",
    description="Remote client for the compliance audit pipeline.",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",  # Example URL
)

print("âœ… RemoteA2aAgent client stub created (conceptual).")


