!pip install -q google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key setup complete.")
except Exception as e:
    print(f"Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google import genai
from google.genai import types


print("ADK components imported successfully.")


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>⚠️ IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, Tuple


from pydantic import BaseModel, Field

class SLAConfig(BaseModel):
    availability: str = Field(description="Availability commitment, e.g., '99.99% uptime for critical services'.")
    latency: str = Field(description="Latency requirements, e.g., 'updates delivered within 5 seconds'.")
    throughput: str = Field(description="The number of requests or operations that the service can process in a given time.")

class NonFunctionalAttributes(BaseModel):
    security: str = Field(description="Security-related NFRs (auth, encryption, IAM, etc.)")
    scalability: str = Field(description="Expected scalability behavior and load characteristics.")
    observability: str = Field(description="Logging, metrics, tracing, monitoring expectations.")
    other: str = Field(description="Other cross-cutting concerns such as reliability or auditability.")

class RequirementsSpec(BaseModel):
    """Structured requirements gathered from the user."""
    domains: List[str] = Field(description="High-level business or functional areas the system must support.")
    user_roles: List[str] = Field(description="Key actors or personas interacting with the system.")
    main_flows: List[str] = Field(description="Core end-to-end business processes or user journeys.")
    slas: SLAConfig = Field(description="Service-level commitments.")
    compliance: List[str] = Field(default_factory=list, description="Regulatory, legal, or policy constraints")
    non_functional: NonFunctionalAttributes = Field(description="Cross-cutting quality attributes.")
    notes: str = Field(description="Assumptions made, ambiguities identified, or open questions requiring clarification.")

class ServiceDefinition(BaseModel):
    name: str = Field(description="Canonical name of the microservice (e.g., OrderService).")
    responsibilities: List[str] = Field(description="List of high-level business or technical capabilities owned by the service.")
    dependencies: List[str] = Field(default_factory=list, description="Names of other services or external systems this service relies on.")
    api_style: str = Field(description="Primary interaction protocol (e.g., 'REST', 'gRPC', 'events').")
    data_store: Optional[str] = Field(description="Persistent storage technology used (e.g., 'PostgreSQL', 'MongoDB'), or null if stateless.")
    cache: Optional[str] = Field(description="In-memory caching solution (e.g., 'Redis'), or null if unused.")

class CommunicationSpec(BaseModel):
    patterns: List[str] = Field(description="Messaging patterns used (e.g., sync-http, async-events)")
    broker: Optional[str] = Field(default=None, description="Message broker (e.g., Kafka, RabbitMQ) if async patterns are used.")
    notes: Optional[str] = Field(default=None, description="Additional clarifications or constraints related to communication")

class DataSpec(BaseModel):
    persistence_approach: str = Field(description="Strategy for data storage boundaries (e.g., per-service schema).")
    multi_tenancy: Optional[str] = Field(default=None, description="Approach for tenant isolation.")
    backup_and_recovery: Optional[str] = Field(default=None, description="Backup strategy, recovery processes, retention policies.")

class SecuritySpec(BaseModel):
    idp: Optional[str] = Field(default=None, description="Identity provider (e.g., Keycloak)")
    authn: Optional[str] = Field(default=None, description="Authentication method (e.g., OIDC/JWT)")
    authz: Optional[str] = Field(default=None, description="Authorization model (e.g., RBAC, ABAC)")
    secrets_management: Optional[str] = Field(default=None, description="Secrets storage (e.g., Vault)")
    network_boundaries: Optional[str] = Field(default=None, description="Segmentation, service mesh, ingress, mTLS, etc.")

class ObservabilitySpec(BaseModel):
    logging: Optional[str] = Field(default=None, description="Logging standards and tools (ELK, Loki, etc.)")
    metrics: Optional[str] = Field(default=None, description="Metrics stack (Prometheus, Grafana), key metrics.")
    tracing: Optional[str] = Field(default=None, description="Distributed tracing stack (Jaeger, Tempo).")
    health_checks: Optional[str] = Field(default=None, description="Liveness/readiness checks strategy.")

class DeploymentSpec(BaseModel):
    orchestrator: Optional[str] = Field(default=None, description="Cluster orchestrator (e.g., Kubernetes).")
    deployment_strategy: Optional[str] = Field(default=None, description="Deployment method (e.g., Blue/Green, Canary).")
    scaling: Optional[str] = Field(default=None, description="HPA/KEDA scaling strategies.")
    environments: Optional[List[str]] = Field(default=None, description="List of deployment environments (dev/staging/prod).")

class ArchitectureDesign(BaseModel):
    services: List["ServiceDefinition"] = Field(description="List of microservices describing the service decomposition.")
    communication: CommunicationSpec = Field(description="Messaging patterns and infrastructure.")
    data: DataSpec = Field(description="Data architecture and persistence model.")
    security: SecuritySpec = Field(description="Identity, authentication, authorization, and secrets.")
    observability: ObservabilitySpec = Field(description="Logging, metrics, tracing, and health-check strategy.")
    deployment: DeploymentSpec = Field(description="Environment orchestration and deployment strategy.")
    rationale: str = Field(description="Key design choices and trade-offs.")

class SolutionForCritic(BaseModel):
    requirements: RequirementsSpec = Field(description="Structured functional and non-functional requirements that the solution must satisfy.")
    architecture: ArchitectureDesign = Field(description="Proposed microservice architecture designed to meet the specified requirements.")

class CriticIssue(BaseModel):
    category: str = Field(description="High-level topic of the issue (e.g., 'security', 'scalability', 'data consistency').")
    description: str = Field(description="Concise explanation of the identified problem or concern.")
    severity: str = Field(description="Impact level - one of 'info' (low), 'warning' (medium), or 'error' (critical).")
    suggestion: str = Field(description="Actionable recommendation to address or mitigate the issue.")

class CriticReview(BaseModel):
    issues: List[CriticIssue] = Field(default_factory=list, description="List of specific, categorized findings from the architectural critique.")
    overall_assessment: str = Field(description="Summary evaluation of the solution’s fitness, strengths, and major weaknesses.")
    score: float = Field(description="Normalized quality rating from 0.0 (unsuitable) to 1.0 (exemplary), reflecting adherence to requirements and best practices.")
    notes: str = Field(description="Additional context, caveats, or open questions not covered in individual issues.")

class DocsBundle(BaseModel):
    """Output of the Docs Agent."""
    overview_md: str = Field(description="High-level system overview: purpose, scope, key design principles, and architecture summary in Markdown format.")
    adrs_md: List[str] = Field(default_factory=list, description="List of Architecture Decision Records (ADRs) in Markdown, each capturing a significant design choice, rationale, alternatives considered, and impact.")



import json
import re

def extract_json(text: str):
    """
    Robust JSON extractor from LLM output.
    """
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        candidate = code_block.group(1)
        return json.loads(candidate)

    curly = re.search(r"(\{.*\})", text, re.DOTALL)
    if curly:
        candidate = curly.group(1)
        return json.loads(candidate)

    raise ValueError("JSON not found in text")


def clean_llm_json(text: str) -> str:
    fenced = re.search(r"```json(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)

    braced = re.search(r"(\{.*\})", text, flags=re.S)
    if braced:
        text = braced.group(1)

    text = text.strip().replace("\u0000", "")

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    if text.count("{") > text.count("}"):
        text += "}" * (text.count("{") - text.count("}"))

    if text.count("[") > text.count("]"):
        text += "]" * (text.count("[") - text.count("]"))

    try:
        json.loads(text)
        return text
    except Exception as e:
        print("JSON after cleaning still invalid:", e)
        print(text)
        raise



from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools.tool_context import ToolContext


GEMINI_MODEL = "gemini-2.0-flash"

def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}


from google.adk.agents import LlmAgent, BaseAgent


requirements_agent_instruction = """
You are a senior software architect acting as a Requirements Engineer.

**Task:**
1. Given a free-text description of a project, extract a *structured* requirements specification for a microservice system.

Return ONLY a valid JSON object.
Wrap nothing in markdown. Do not add commentary.
If unsure, output an empty list or empty string placeholders, but keep JSON valid.
""".strip()
requirements_agent_description = "An AI agent that transforms free-text project descriptions into structured microservice requirements in standardized JSON format."
requirements_agent_output_key = "requirements_doc"

requirements_agent = LlmAgent(
    name="requirements_agent",
    model=GEMINI_MODEL,
    include_contents='none',
    output_schema=RequirementsSpec,
    instruction=requirements_agent_instruction,
    description=requirements_agent_description,
    output_key=requirements_agent_output_key
)


def create_architecture_agent(agent_name: str, instruction: str, tools: List[Any] = []) -> LlmAgent:
    architecture_agent_description = "An AI agent that designs a production-ready microservice architecture from structured requirements (and optional feedback), outputting a detailed, schema-compliant JSON covering services, communication, data, security, observability, and deployment—plus architectural rationale."
    architecture_agent_output_key = "current_architecture_doc"
    return LlmAgent(
        name=agent_name,
        model=GEMINI_MODEL,
        include_contents='none',
        output_schema=ArchitectureDesign,
        instruction=instruction,
        description=architecture_agent_description,
        output_key=architecture_agent_output_key,
        tools=tools
    )

initial_architecture_agent_instruction = """
You are a principal cloud/solution architect.
**Structured requirements specification:**
```
{{requirements_doc}}
```

**Task:**
1. Analyze the 'Structured requirements specification' and design a pragmatic microservices architecture that meets the requirements.

Return ONLY a valid JSON object.
Wrap nothing in markdown. Do not add commentary.
If unsure, output an empty list or empty string placeholders, but keep JSON valid.
"""
architecture_agent = create_architecture_agent("architecture_agent", initial_architecture_agent_instruction)


critic_agent_instruction = """
You are a strict, production-focused Architecture Review Board.

**Structured requirements specification:**
```
{{requirements_doc}}
```

**Proposed architectural design:**
```
{{current_architecture_doc}}
```

**Task:**
1. Analyze the 'Structured requirements specification' and 'Proposed architectural design', perform a thorough checklist-based review covering:
  - Fault tolerance (retries, timeouts, circuit breakers, bulkheads)
  - Resilience & scalability
  - Observability (logging, metrics, tracing, alerting)
  - Security (least privilege, secrets, network segmentation)
  - Operational readiness (health checks, deployments, rollbacks, data migrations)

Return ONLY a valid JSON object.
Wrap nothing in markdown. Do not add commentary.
If unsure, output an empty list or empty string placeholders, but keep JSON valid.
""".strip()
critic_agent_description = ""
critic_agent_output_key = "critic_review"

critic_agent_in_loop = LlmAgent(
    name="critic_agent_in_loop",
    model=GEMINI_MODEL,
    include_contents='none',
    output_schema=CriticReview,
    instruction=critic_agent_instruction,
    description=critic_agent_description,
    output_key=critic_agent_output_key
)


doc_agent_instruction = """
You are a senior architect who writes clear architecture docs.

**Final architectural design:**
```
{{current_architecture_doc}}
```

**Critic review:**
```
{{critic_review}}
```

**Task:**
1. Given the 'Final architectural design', 'Critic review' and produce:
  - A concise architecture overview in Markdown.
  - A list of ADRs (Architecture Decision Records) in Markdown. Each ADR should include:
    - Context.
    - Decision.
    - Consequences.
    - Alternatives (including rejected ones).

Return ONLY a valid JSON object.
Wrap nothing in markdown. Do not add commentary.
If unsure, output an empty list or empty string placeholders, but keep JSON valid.
""".strip()
doc_agent_description = ""
doc_agent_output_key = "documentation_for_architecture"

doc_agent = LlmAgent(
    name="doc_agent",
    model=GEMINI_MODEL,
    include_contents='none',
    output_schema=DocsBundle,
    instruction=doc_agent_instruction,
    description=doc_agent_description,
    output_key=doc_agent_output_key
)


from google.adk.agents import LoopAgent


MIN_CRITIC_SCORE = 0.7
NO_MAJOR_ISSUES_FOUND = "No major issues found."
SOME_SERIOUS_ISSUES_FOUND = "There are some serious issues."

def critic_score_check(critic_review: CriticReview, tool_context: ToolContext) -> str:
    """
    A function for checking the critic's numerical rating.
    If the rating is greater than the minimum value, we exit the "architecture-criticism" cycle.
    """
    if not critic_review:
        return {"status": "success", "result": COMPLETION_PHRASE, "score": 0}

    score = getattr(critic_review, "score", 0.0)
    if score >= MIN_CRITIC_SCORE:
        tool_context.actions.transfer_to_agent = "doc_agent"
        return {"status": "success", "result": COMPLETION_PHRASE, "score": score}

    return {"status": "failed", "result": SOME_SERIOUS_ISSUES_FOUND, "score": score}

critic_metrics_agent_in_loop_instruction = """
You are an automated Critic metrics evaluator.

**Your only task is:**
1. Accept a CriticReview JSON object.
2. Pass it as is to the 'critic_score_check' function.
3. Return only the raw output of the tool without modifications, explanations, or additional text.

Do not interpret, validate, or modify the input data.
Do not generate new content.
Execute the tool call and pass its output exactly as is.
"""
critic_metrics_agent_in_loop_description = """A lightweight, deterministic agent that computes quantitative scores using 
the 'critic_score_check' function based on the output of an architectural critic."""
critic_metrics_agent_in_loop_output_key = "critic_score_check_response"

critic_metrics_agent_in_loop = LlmAgent(
    name="critic_metrics_agent_in_loop",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=critic_metrics_agent_in_loop_instruction,
    description=critic_metrics_agent_in_loop_description,
    output_key=critic_metrics_agent_in_loop_output_key,
    tools=[critic_score_check]
)


loop_architecture_agent_instruction = """
You are the lead cloud architect who iteratively refines the microservices architecture based on criticism and feedback,
improving design decisions, closing gaps, and maintaining compliance until the solution is stable and ready for documentation.

**Current architectural design:**
```
{{current_architecture_doc}}
```

**Critic review:**
```
{{critic_review}}
```

**Critic's score check result:**
```
{{critic_score_check_response.result}}
```

**Task:**
Analyze the 'Critic's score check result'.
IF the critique is *exactly* "No major issues found.": You MUST call the 'exit_loop' function. Do not output any text.
ELSE (the critique contains actionable feedback):
Carefully apply the suggestions to improve the 'Current architectural design'.
"""
architecture_agent_in_loop = create_architecture_agent("architecture_agent_in_loop", loop_architecture_agent_instruction, [])

architectural_design_loop = LoopAgent(
    name="architectural_design_loop",
    sub_agents=[
        critic_agent_in_loop,
        critic_metrics_agent_in_loop,
        architecture_agent_in_loop,
    ],
    max_iterations=5 # Limit loops
)


from google.adk.agents import SequentialAgent

root_agent = None
root_agent = SequentialAgent(
    name="architectural_design_pipeline",
    sub_agents=[
        requirements_agent,         # Run first to create requirements
        architecture_agent,         # Then first architecture
        architectural_design_loop,  # Then run the critic->architectural_design_loop
        doc_agent                   # In conclusion, the documentation phase.
    ],
    description="Orchestrates the end-to-end solution engineering workflow: starts with requirements extraction, iterates on architectural design via critique-driven refinement, and concludes with the documentation phase - producing and progressively improving the final document until the exit condition is met."
)


from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

USER_ID = "Test user"
SESSION_ID = "test_user_1"
APP_NAME = "architectural_design_pipeline_app"

async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner

async def call_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    result = []
    async for event in events:
        result.append(event)

    return result

user_query = """
We need to design a microservice architecture for a logistics platform that
handles orders, routing, and real-time tracking for a large retail company.
We care a lot about reliability, auditability, and EU data protection.
"""
agent_events = await call_agent_async(user_query)


def find_last_event_by_author(events, author_name):
    for event in reversed(events):
        if getattr(event, 'author', None) == author_name:
            return event
    return None

requirements_event = find_last_event_by_author(agent_events, "requirements_agent")
architecture_event = find_last_event_by_author(agent_events, "architecture_agent")
architecture_in_loop_event = find_last_event_by_author(agent_events, "architecture_agent_in_loop")
architecture_event = architecture_in_loop_event if architecture_in_loop_event is not None else architecture_event
critic_agent_event = find_last_event_by_author(agent_events, "critic_agent_in_loop")
doc_event = find_last_event_by_author(agent_events, "doc_agent")


requirements_dict = extract_json(requirements_event.content.parts[0].text)
requirements_spec = RequirementsSpec(**requirements_dict)

architecture_dict = extract_json(architecture_event.content.parts[0].text)
architecture_design = ArchitectureDesign(**architecture_dict)

critic_review_dict = extract_json(critic_agent_event.content.parts[0].text)
critic_review = CriticReview(**critic_review_dict)

doc_dict = extract_json(doc_event.content.parts[0].text)
doc_bundle = DocsBundle(**doc_dict)


from pydantic import BaseModel
from typing import List, Optional

CRITIC_CHECKLIST = {
    "fault_tolerance": [
        "retry", "timeout", "circuit breaker", "bulkhead"
    ],
    "observability": [
        "centralized logging", "metrics", "tracing", "correlation id"
    ],
    "security": [
        "least privilege", "secrets management", "network boundaries"
    ],
    "operations": [
        "health checks", "readiness", "rollbacks"
    ],
}

class EvaluationResult(BaseModel):
    checklist_coverage: Dict[str, float]
    average_coverage: float
    missing_topics: Dict[str, List[str]]


def evaluate_critic_against_checklist(review: CriticReview) -> EvaluationResult:
    """
    Simple heuristic eval:
    - Concatenate critic overall assessment + all issue descriptions/suggestions.
    - For each checklist category, count how many required keywords are mentioned.
    """
    text_parts: List[str] = [review.overall_assessment, review.notes]
    for issue in review.issues:
        text_parts.append(issue.description)
        text_parts.append(issue.suggestion)
    text = " ".join(text_parts).lower()

    coverage: Dict[str, float] = {}
    missing: Dict[str, List[str]] = {}

    for category, keywords in CRITIC_CHECKLIST.items():
        hits = sum(1 for kw in keywords if kw.lower() in text)
        coverage[category] = hits / max(len(keywords), 1)
        missing[category] = [kw for kw in keywords if kw.lower() not in text]

    avg = sum(coverage.values()) / max(len(coverage), 1)
    return EvaluationResult(
        checklist_coverage=coverage,
        average_coverage=avg,
        missing_topics=missing,
    )


eval_result = evaluate_critic_against_checklist(critic_review)
print("### Checklist coverage (per category)")
for category, value in eval_result.checklist_coverage.items():
    print(f"- **{category}**: {value:.2f}")

print("\n### Missing topics")
if eval_result.missing_topics:
    for category, items in eval_result.missing_topics.items():
        print(f"- {category}:")
        for item in items:
            print(f"   - {item}")
else:
    print("No missing topics!")

print("\n### Average coverage")
print(f"{eval_result.average_coverage:.3f}")

print("\n### Final score interpretation")
if eval_result.average_coverage >= 0.9:
    print("Excellent architecture — almost no missing elements.")
elif eval_result.average_coverage >= 0.75:
    print("Very good — a few areas to refine.")
elif eval_result.average_coverage >= 0.5:
    print("Medium quality — noticeable gaps in architecture.")
else:
    print("Poor — needs significant improvement.")


def print_requirements(requirements: RequirementsSpec):
    print("## Requirements")

    print("### Domains")
    print("\n - ".join(requirements.domains))

    print("### User roles")
    print("\n - ".join(requirements.user_roles))

    print("### Main flows")
    print("\n - ".join(requirements.main_flows))

    print("### SLA")
    print("#### Availability")
    print(requirements.slas.availability)
    print("#### Latency")
    print(requirements.slas.latency)
    print("#### Throughput")
    print(requirements.slas.throughput)

    print("### Compliance")
    print("\n - ".join(requirements.compliance))

    print("### Non functional")
    print("#### Security")
    print(requirements.non_functional.security)
    print("#### Scalability")
    print(requirements.non_functional.scalability)
    print("#### Observability")
    print(requirements.non_functional.observability)
    print("#### Other")
    print(requirements.non_functional.other)

    print("### Notes")
    print(requirements.notes)
    print("\n")


def print_architecture(architecture: ArchitectureDesign):
    print("## Architecture")
    print_services(architecture.services)
    print_communication(architecture.communication)
    print_data(architecture.data)
    print_security(architecture.security)
    print_observability(architecture.observability)
    print_deployment(architecture.deployment)
    print_rationale(architecture.rationale)
    print("\n")

def print_services(services: List[ServiceDefinition]):
    print("### Services")
    rows = []
    for s in services:
        print(f"#### {s.name}")
        deps = ", ".join(s.dependencies) if s.dependencies else ""
        resp = ", ".join(s.responsibilities) if s.responsibilities else ""
        print(f"   Responsibilities: {resp}")
        print(f"   Dependencies: {deps}")
        print(f"   API Style: {s.api_style}")
        print(f"   Data store: {s.data_store}")
        print(f"   Cache: {s.cache}")

def print_communication(communication: CommunicationSpec):
    print("### Communication")
    patterns = ", ".join(communication.patterns)
    print(f"   Patterns: {patterns}")
    print(f"   Broker: {communication.broker}")
    print(f"   Notes: {communication.notes}")

def print_data(data: DataSpec):
    print("### Data")
    print(f"   Persistence: {data.persistence_approach}")
    print(f"   Multi tenancy: {data.multi_tenancy}")
    print(f"   Backup and recovery: {data.backup_and_recovery}")

def print_security(sec: SecuritySpec):
    print("### Security")
    print(f"   IdP: {sec.idp}")
    print(f"   Authentication: {sec.authn}")
    print(f"   Authorization: {sec.authz}")
    print(f"   Secrets management: {sec.secrets_management}")
    print(f"   Network boundaries: {sec.network_boundaries}")

def print_observability(data: ObservabilitySpec):
    print("### Observability")
    print(f"   Logging: {data.logging}")
    print(f"   Metrics: {data.metrics}")
    print(f"   Tracing: {data.tracing}")
    print(f"   Health checks: {data.health_checks}")

def print_deployment(data: DeploymentSpec):
    print("### Deployment")
    print(f"   Orchestrator: {data.orchestrator}")
    print(f"   Deployment strategy: {data.deployment_strategy}")
    print(f"   Scaling: {data.scaling}")
    print(f"   Environments: {data.environments}")

def print_rationale(data: str):
    print("### Rationale")
    print(data)


def print_critic_review(critic_review: CriticReview):
    print("## Critic review")
    print_score(critic_review.score)
    print_issues(critic_review.issues)
    print_overall_assessment(critic_review.overall_assessment)
    print_notes(critic_review.notes)
    print("\n")

def print_score(score: float):
    print("### Score")
    print(score)

def print_issues(issues: List[CriticIssue]):
    print("### Issues")
    rows = []
    for s in issues:
        print(f"#### {s.category}")
        print(f"   Description: {s.description}")
        print(f"   Severity: {s.severity}")
        print(f"   Suggestion: {s.suggestion}")

def print_overall_assessment(overall_assessment: str):
    print("### Overall assessment")
    print(overall_assessment)

def print_notes(data: str):
    print("### Notes")
    print(data)


def print_docs_bundle(docs_bundle: DocsBundle):
    print("## Documentation")
    print("### Overview")
    print(docs_bundle.overview_md)
    print("### ADR's")
    adrs = "\n   -".join([a for a in docs_bundle.adrs_md])
    print(adrs)
    print("\n")


def pretty_print(user_query: str,
                 requirements: RequirementsSpec,
                 architecture: ArchitectureDesign,
                 critic_review: CriticReview,
                 doc_bundle: DocsBundle):
    print("# Architectural solution")
    print("\n")

    # User problem
    print("## User problem")
    print(user_query)
    print("\n")

    # Requirements
    print_requirements(requirements)

    # Architecture
    print_architecture(architecture)

    # Critic review
    print_critic_review(critic_review)

    # Documentation
    print_docs_bundle(doc_bundle)

pretty_print(user_query, requirements_spec, architecture_design, critic_review, doc_bundle)

