"""
Standalone mock of the "AI Agents Capstone Starter" with Jupyter-safe runner.

This version avoids ImportError from google.adk and also handles the
"asyncio.run() cannot be called from a running event loop" error that occurs
in Jupyter/Colab by using a fallback that schedules the coroutine and waits
for completion.

Run:
    python capstone_starter_fixed.py
or in Jupyter/Colab, paste into a cell and run the cell directly.
"""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional, List, Callable

# -------------------------
# Simple "tool" decorator
# -------------------------
def tool(fn: Callable):
    """Mark a function as a tool (no-op wrapper for demonstration)."""
    fn._is_tool = True
    return fn


# -------------------------
# Event & Trace dataclasses
# -------------------------
@dataclass
class AgentTrace:
    agent_name: str


@dataclass
class ToolRequest:
    name: str


@dataclass
class ToolResponse:
    output: str


@dataclass
class StreamEvent:
    event_type: str
    text: Optional[str] = None
    agent_trace: Optional[AgentTrace] = None
    tool_request: Optional[ToolRequest] = None
    tool_response: Optional[ToolResponse] = None


# -------------------------
# Tools
# -------------------------
@tool
def simple_research_tool(query: str) -> str:
    """
    Mock research tool. Replace with real API calls in your project.
    """
    print(f"\n--- TOOL CALLED: simple_research_tool ---")
    print(f"--- Query: {query} ---")
    mock_data = (
        f"Mock data for '{query}': AI agents are systems that perceive their environment, "
        "make decisions, and take actions. The ADK is a kit for building them."
    )
    print(f"--- Returning: {mock_data} ---")
    return mock_data


# -------------------------
# Agent classes (mocks)
# -------------------------
class LlmAgent:
    def __init__(self, name: str, model: str, instruction: str, tools: Optional[List[Callable]] = None):
        self.name = name
        self.model = model
        self.instruction = instruction
        self.tools = tools or []

    async def run(self, input_text: str):
        """
        Mock run method. For ResearcherAgent we call the first tool (if any).
        For WriterAgent we summarize input_text.
        This yields StreamEvent objects to simulate streaming behavior.
        """
        # Agent start
        yield StreamEvent(event_type="agent_start", agent_trace=AgentTrace(agent_name=self.name))
        await asyncio.sleep(0)  # yield control

        # If tools exist, "request" tool and yield events
        if self.tools:
            tool_fn = self.tools[0]
            yield StreamEvent(event_type="tool_request", tool_request=ToolRequest(name=tool_fn.__name__))
            # simulate tool processing time
            await asyncio.sleep(0.2)
            tool_out = tool_fn(input_text)
            yield StreamEvent(event_type="tool_response", tool_response=ToolResponse(output=tool_out))
            # Provide the tool output as text so downstream agent can use it
            yield StreamEvent(event_type="text", text=tool_out, agent_trace=AgentTrace(agent_name=self.name))
        else:
            # Writer agent: create a two-paragraph summary
            await asyncio.sleep(0.1)
            # Keep first paragraph as a short extracted intro (up to 300 chars)
            intro = input_text.strip().replace("\n", " ")[:300]
            p1 = intro if intro else "Research provided no content to summarize."
            p2 = (
                "In summary, AI agents combine perception, decision-making, and action. "
                "Toolkits like ADK provide scaffolding to build and orchestrate them."
            )
            summary = f"{p1}\n\n{p2}"
            yield StreamEvent(event_type="text", text=summary, agent_trace=AgentTrace(agent_name=self.name))

        # Agent end
        yield StreamEvent(event_type="agent_end", agent_trace=AgentTrace(agent_name=self.name))


class SequentialAgent:
    """
    Runs a sequence of sub-agents in order.
    """
    def __init__(self, name: str, sub_agents: List[LlmAgent]):
        self.name = name
        self.sub_agents = sub_agents


# -------------------------
# Session service mock
# -------------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    async def create_session(self, user_id: str, agent_name: str):
        sess_id = str(uuid.uuid4())
        session = {"id": sess_id, "user_id": user_id, "agent_name": agent_name}
        self.sessions[sess_id] = session
        return session


# -------------------------
# App registry & App mock
# -------------------------
class MockApp:
    def __init__(self, session_service: InMemorySessionService, workflow: SequentialAgent):
        self.session_service = session_service
        self.workflow = workflow

    async def async_create_session(self, user_id: str, agent_name: str):
        return await self.session_service.create_session(user_id=user_id, agent_name=agent_name)

    async def async_stream_query(self, agent_name: str, user_id: str, session_id: str, message: str):
        """
        Simulate streaming: call each agent sequentially and yield StreamEvent objects.
        researcher -> writer
        """
        intermediate_text = message
        for agent in self.workflow.sub_agents:
            # agent.run yields events
            async for ev in agent.run(intermediate_text):
                # For tool_response events, capture tool output to pass to next agent
                yield ev
                if ev.tool_response:
                    intermediate_text = ev.tool_response.output
                if ev.text and not agent.tools:
                    # Writer produced final text; set as intermediate_text
                    intermediate_text = ev.text
            # small pause between agents to simulate processing
            await asyncio.sleep(0.05)
        # Final answer event
        yield StreamEvent(event_type="final_answer", text=intermediate_text)


class AppRegistry:
    def __init__(self):
        self._apps = {}

    def get_app(self, services: dict, workflow: SequentialAgent):
        session_service = services.get(InMemorySessionService)
        if session_service is None:
            session_service = InMemorySessionService()
        app = MockApp(session_service=session_service, workflow=workflow)
        return app


app_registry = AppRegistry()

# -------------------------
# Build the agents & workflow
# -------------------------
researcher_agent = LlmAgent(
    name="ResearcherAgent",
    model="mock-model",
    instruction=(
        "You are an expert researcher whose job is to use the simple_research_tool "
        "to find information based on the user's request."
    ),
    tools=[simple_research_tool],
)

writer_agent = LlmAgent(
    name="WriterAgent",
    model="mock-model",
    instruction=(
        "You are an expert content writer. Given a block of research text, "
        "write a clean, concise, 2-paragraph summary."
    ),
    tools=[],  # no tools
)

research_workflow = SequentialAgent(
    name="ResearchWorkflow",
    sub_agents=[researcher_agent, writer_agent],
)

SESSION_SERVICE = InMemorySessionService()

# -------------------------
# Main async runner
# -------------------------
async def main():
    print("==========================================")
    print("  AI Agents Capstone Starter (Mock)       ")
    print("==========================================")

    # Create the mock app, providing the session service
    app = app_registry.get_app(
        services={InMemorySessionService: SESSION_SERVICE},
        workflow=research_workflow,
    )

    print("\nStarting a new conversation session...")
    user_id = "capstone-user-123"
    session = await app.async_create_session(user_id=user_id, agent_name=research_workflow.name)
    session_id = session["id"]
    print(f"Session created: {session_id}")

    prompt = "What are AI Agents and the ADK?"
    print(f"\nUser: {prompt}\n")
    print("--- Agent Workflow Starting ---")

    final_response = ""
    try:
        async for event in app.async_stream_query(
            agent_name=research_workflow.name,
            user_id=user_id,
            session_id=session_id,
            message=prompt,
        ):
            print(f"--- Event: {event.event_type} ---")
            if event.agent_trace:
                print(f"[Trace: {event.agent_trace.agent_name} is working...]")
            if event.tool_request:
                print(f"[Trace: Requesting tool {event.tool_request.name}]")
            if event.tool_response:
                print(f"[Trace: Got tool response]")
            if event.text:
                # print text; add to final_response
                print(event.text)
                final_response += (event.text + "\n")
    except Exception as e:
        print("\n--- ERROR ---")
        print(e)

    print("\n--- Agent Workflow Complete ---")
    print("\n==========================================")
    print("           Final Response:")
    print("==========================================")
    print(final_response.strip())


# -------------------------
# Runner that is safe for both script and Jupyter/Colab
# -------------------------
def run_main_safely():
    """
    Try to run with asyncio.run(). If a running loop is present (Jupyter/Colab),
    schedule the coroutine and wait for completion via a simple blocking loop.
    This avoids calling top-level `await` or requiring extra packages.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. This mock does not need it.")

    try:
        # Preferred for normal Python interpreter
        asyncio.run(main())
    except RuntimeError:
        # A running loop exists (e.g., Jupyter). Schedule main() and wait.
        loop = asyncio.get_event_loop()
        task = loop.create_task(main())

        # Busy-wait until the task completes. This is simple and works in notebooks.
        try:
            while not task.done():
                time.sleep(0.05)
        except KeyboardInterrupt:
            # allow user to interrupt
            task.cancel()
            raise

        # If the task raised, re-raise exception here
        if task.cancelled():
            raise RuntimeError("Task was cancelled.")
        if task.exception():
            raise task.exception()


if __name__ == "__main__":
    run_main_safely()





