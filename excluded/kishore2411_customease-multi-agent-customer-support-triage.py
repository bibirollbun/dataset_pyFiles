import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.adk.tools import AgentTool, FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.genai import types

print("âœ… ADK components imported successfully.")


# âœ… Proper Gemini client for direct LLM calls (outside ADK agents)
from google import genai

genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("âœ… google.genai client initialized.")


# ----------------------------------------
# Initialize session, memory and LLM model
# ----------------------------------------

# InMemory session + memory store (for storing conversation history / triage states)
session_service = InMemorySessionService()  
memory_service = InMemoryMemoryService()    

# Initialize Gemini LLM wrapper from Google ADK
llm = Gemini(model="gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"])

print("âœ… Session, Memory & LLM initialized successfully.")


from typing import Dict, Any, List

def log_event(event_type: str, data: Dict[str, Any]):
    """Structured logging tool to capture observability for the pipeline."""
    print(f"[LOG] {event_type} -> {data}")


def classify_intent_llm(message: str) -> str:
    """LLM-powered classification tool: determine ticket intent category."""
    prompt = f"""
    You are an expert customer support triage classifier.
    Classify the following message into one intent category:
    - billing
    - technical
    - general

    Message:
    {message}

    Respond with only the category name.
    """
    response = genai_client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip().lower()


def estimate_urgency_llm(message: str) -> str:
    """LLM urgency scoring tool."""
    prompt = f"""
    You are an urgency estimator for customer support.
    Label the urgency as exactly one of: high, medium, low.

    Message:
    {message}

    Respond with only one word.
    """
    response = genai_client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip().lower()


# Lightweight Knowledge Base (expandable)
FAQ_KB = [
    {"id": "faq1", "title": "Refund Policy", "content": "We offer refunds within 30 days of purchase. To request a refund, contact support with your order ID.", "tags": ["billing", "refund", "charge"]},
    {"id": "faq2", "title": "Password Reset", "content": "You can reset your password by clicking 'Forgot Password' on the login page and following the instructions.", "tags": ["password", "reset", "account", "login"]},
    {"id": "faq3", "title": "App Crashing", "content": "If the app is crashing, please try reinstalling it or clearing the app cache and data.", "tags": ["bug", "crash", "technical", "error"]},
]


def faq_lookup_tool(message: str, top_k: int = 2):
    """Simple keyword-based FAQ search."""
    msg = message.lower()
    results = []
    for faq in FAQ_KB:
        score = sum(1 for tag in faq["tags"] if tag in msg)
        if score > 0:
            results.append({**faq, "score": score})
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


print("âœ… Tools (LLM + FAQ + logger) defined.")


from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SupportTicketInput:
    ticket_id: str
    customer_id: str
    message: str
    channel: str = "email"  # or "chat", etc.


@dataclass
class TriageDecision:
    intent: str          # "billing", "technical", "general"
    urgency: str         # "low", "medium", "high"
    team: str            # e.g. "Billing Team"
    confidence: float


@dataclass
class KnowledgeSnippet:
    id: str
    title: str
    content: str
    score: float


@dataclass
class ResponseDraft:
    ticket_id: str
    text: str
    triage_decision: TriageDecision
    used_snippets: List[KnowledgeSnippet]


@dataclass
class CustomerSessionState:
    customer_id: str
    history: List[str] = field(default_factory=list)
    summary: str = ""
    last_triage: Optional[TriageDecision] = None

print("âœ… Data models defined.")


class SessionMemory:
    def __init__(self):
        self.sessions: Dict[str, CustomerSessionState] = {}

    def get_or_create(self, customer_id: str) -> CustomerSessionState:
        if customer_id not in self.sessions:
            self.sessions[customer_id] = CustomerSessionState(customer_id=customer_id)
        return self.sessions[customer_id]

    def add_message(self, customer_id: str, message: str):
        session = self.get_or_create(customer_id)
        session.history.append(message)

    def set_summary(self, customer_id: str, summary: str):
        session = self.get_or_create(customer_id)
        session.summary = summary

    def set_last_triage(self, customer_id: str, triage: TriageDecision):
        session = self.get_or_create(customer_id)
        session.last_triage = triage


# Create a global session memory instance for the orchestrator
session_memory = SessionMemory()

print("âœ… Session memory initialized.")


def summarize_session_history(customer_id: str):
    """Use the LLM to generate a short summary of the customer's conversation so far."""
    session = session_memory.get_or_create(customer_id)
    if not session.history:
        return

    joined = "\n".join(session.history[-5:])  # last few messages

    prompt = f"""
    You are a summarization assistant.
    Summarize the following customer conversation into 3â€“4 lines, focusing on:
    - Main issue
    - Any important constraints
    - Any steps already tried (if mentioned)

    Conversation:
    {joined}
    """

    try:
        response = genai_client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        summary = response.text.strip()
        session_memory.set_summary(customer_id, summary)
        log_event("context_summary_updated", {
            "customer_id": customer_id,
            "summary": summary
        })
    except Exception as e:
        log_event("context_summary_error", {
            "customer_id": customer_id,
            "error": str(e)
        })

print("âœ… Context summarization function defined.")


import uuid

class BaseAgent:
    def __init__(self, name: str, memory: SessionMemory):
        self.name = name
        self.memory = memory

    def run(self, *args, **kwargs):
        raise NotImplementedError


class IntakeAgent(BaseAgent):
    def run(self, ticket: SupportTicketInput) -> SupportTicketInput:
        log_event("agent_intake_start", {"ticket_id": ticket.ticket_id})
        # Store raw message in session history
        self.memory.add_message(ticket.customer_id, ticket.message)
        # Update conversation summary (context compaction)
        summarize_session_history(ticket.customer_id)
        log_event("agent_intake_end", {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id
        })
        return ticket


class TriageAgent(BaseAgent):
    def run(self, ticket: SupportTicketInput) -> TriageDecision:
        log_event("agent_triage_start", {"ticket_id": ticket.ticket_id})

        intent = classify_intent_llm(ticket.message)
        urgency = estimate_urgency_llm(ticket.message)

        team_map = {
            "billing": "Billing Team",
            "technical": "Technical Support",
            "general": "Customer Care"
        }
        team = team_map.get(intent, "Customer Care")

        decision = TriageDecision(
            intent=intent,
            urgency=urgency,
            team=team,
            confidence=0.95
        )

        self.memory.set_last_triage(ticket.customer_id, decision)
        log_event("agent_triage_end", {
            "ticket_id": ticket.ticket_id,
            "decision": decision.__dict__
        })
        return decision


class KnowledgeAgent(BaseAgent):
    def run(self, ticket: SupportTicketInput, triage: TriageDecision) -> List[KnowledgeSnippet]:
        log_event("agent_knowledge_start", {"ticket_id": ticket.ticket_id})
        kb_hits = faq_lookup_tool(ticket.message)
        snippets = [
            KnowledgeSnippet(
                id=item["id"],
                title=item["title"],
                content=item["content"],
                score=item["score"],
            )
            for item in kb_hits
        ]
        log_event("agent_knowledge_end", {
            "ticket_id": ticket.ticket_id,
            "snippets": [s.id for s in snippets],
        })
        return snippets


class ResponseAgent(BaseAgent):
    def run(
        self,
        ticket: SupportTicketInput,
        triage: TriageDecision,
        snippets: List[KnowledgeSnippet],
    ) -> ResponseDraft:
        log_event("agent_response_start", {"ticket_id": ticket.ticket_id})

        kb_bullets = "\n".join(
            [f"- {s.title}: {s.content}" for s in snippets]
        ) or "No specific FAQ entries were matched, but our team will review your case in detail."

        session = self.memory.get_or_create(ticket.customer_id)
        summary_text = session.summary or "No previous conversation context."

        prompt = f"""
        You are a polite and helpful customer support agent.

        TASK:
        Write a clear, empathetic, and professional reply to the customer.

        INFORMATION:
        - Customer message: {ticket.message}
        - Triage intent: {triage.intent}
        - Triage urgency: {triage.urgency}
        - Routed team: {triage.team}
        - Conversation summary: {summary_text}
        - Relevant knowledge base entries:
        {kb_bullets}

        REQUIREMENTS:
        - Acknowledge the user's issue.
        - Use the knowledge base information if relevant.
        - Mention that the ticket is routed to the appropriate team.
        - Keep it within 2â€“3 short paragraphs.
        - Maintain a friendly and professional tone.

        Now write the response email.
        """

        try:
            response_llm = genai_client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )
            text = response_llm.text.strip()
        except Exception as e:
            log_event("agent_response_llm_error", {
                "ticket_id": ticket.ticket_id,
                "error": str(e)
            })
            # Fallback template
            text = (
                f"Hello,\n\n"
                f"Thank you for reaching out regarding your **{triage.intent}** issue.\n\n"
                f"{kb_bullets}\n\n"
                f"Your request has been routed to our **{triage.team}** with **{triage.urgency}** priority.\n"
                f"We will get back to you as soon as possible.\n\n"
                f"Best regards,\nCustomer Support"
            )

        response = ResponseDraft(
            ticket_id=ticket.ticket_id,
            text=text,
            triage_decision=triage,
            used_snippets=snippets,
        )

        log_event("agent_response_end", {"ticket_id": ticket.ticket_id})
        return response


class ReviewerAgent(BaseAgent):
    def run(self, response: ResponseDraft) -> bool:
        log_event("agent_reviewer_start", {"ticket_id": response.ticket_id})
        ok = (
            "Thank you" in response.text
            and len(response.text) > 50
        )
        log_event("agent_reviewer_end", {
            "ticket_id": response.ticket_id,
            "approved": ok,
        })
        return ok


class CustomeaseOrchestrator:
    def __init__(self, memory: SessionMemory):
        self.memory = memory
        self.intake_agent = IntakeAgent("IntakeAgent", memory)
        self.triage_agent = TriageAgent("TriageAgent", memory)
        self.knowledge_agent = KnowledgeAgent("KnowledgeAgent", memory)
        self.response_agent = ResponseAgent("ResponseAgent", memory)
        self.reviewer_agent = ReviewerAgent("ReviewerAgent", memory)

    def handle_ticket(self, ticket: SupportTicketInput) -> ResponseDraft:
        ticket = self.intake_agent.run(ticket)
        triage = self.triage_agent.run(ticket)
        snippets = self.knowledge_agent.run(ticket, triage)
        response = self.response_agent.run(ticket, triage, snippets)

        approved = self.reviewer_agent.run(response)
        if not approved:
            log_event("orchestrator_retry", {"ticket_id": ticket.ticket_id})
            response.text += "\n\n[Note: This response was flagged for review.]"

        return response

print("âœ… Agents and orchestrator defined.")


# Create orchestrator instance
orchestrator = CustomeaseOrchestrator(session_memory)

test_tickets = [
    SupportTicketInput(
        ticket_id=str(uuid.uuid4()),
        customer_id="cust_1",
        message="I was charged twice for my monthly subscription and I need a refund urgently."
    ),
    SupportTicketInput(
        ticket_id=str(uuid.uuid4()),
        customer_id="cust_2",
        message="The app keeps crashing every time I try to upload a file. Please fix this issue."
    ),
    SupportTicketInput(
        ticket_id=str(uuid.uuid4()),
        customer_id="cust_3",
        message="Hi, I forgot my password. How can I reset it and log in again?"
    ),
]

results = []

for t in test_tickets:
    print("=" * 100)
    print("TICKET MESSAGE:\n", t.message)
    response = orchestrator.handle_ticket(t)
    triage = response.triage_decision
    print("\nTRIAGE DECISION:")
    print(f"  - Intent:   {triage.intent}")
    print(f"  - Urgency:  {triage.urgency}")
    print(f"  - Team:     {triage.team}")
    print("\nGENERATED RESPONSE:\n")
    print(response.text)
    results.append(response)

print("\nâœ… Demo run complete.")


expected_labels = [
    {"intent": "billing", "team": "Billing Team", "urgency": "high"},
    {"intent": "technical", "team": "Technical Support", "urgency": "medium"},
    {"intent": "general", "team": "Customer Care", "urgency": "low"},
]

intent_correct = 0
team_correct = 0
urgency_correct = 0

for resp, expected in zip(results, expected_labels):
    triage = resp.triage_decision
    if triage.intent == expected["intent"]:
        intent_correct += 1
    if triage.team == expected["team"]:
        team_correct += 1
    if triage.urgency == expected["urgency"]:
        urgency_correct += 1

n = len(expected_labels)
print("ğŸ“Š Evaluation Summary")
print(f"- Intent accuracy:  {intent_correct}/{n}")
print(f"- Team routing:     {team_correct}/{n}")
print(f"- Urgency accuracy: {urgency_correct}/{n}")




