import os
from textwrap import dedent
import asyncio

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.genai.errors import ClientError

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

BASE_MODEL = "gemini-2.0-flash"
APP_NAME = "story_debate_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session"



def create_story_agents():
    story_requirements_agent = LlmAgent(
        name="StoryRequirementsAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Extract a structured story specification from the user's prompt.
        Output JSON:
        { "plot_type": "...", "genre": "...", "tone": "...",
          "target_length_words": 2000, "audience": "...", "extra_constraints": "..." }
        """),
        output_key="story_spec_json",
    )

    plot_agent = LlmAgent(
        name="PlotArchitectAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Using {story_spec_json}, produce a 3-5 act plot outline.
        Output JSON: { "acts": [ { "act_name": "...", "act_summary": "...", "key_beats": [...] } ] }
        """),
        output_key="plot_outline_json",
    )

    world_agent = LlmAgent(
        name="WorldBuilderAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Using {story_spec_json} and {plot_outline_json}, build the world.
        Output JSON: { "setting_overview": "...", "rules_and_constraints": "...", "key_locations": [...] }
        """),
        output_key="world_json",
    )

    character_agent = LlmAgent(
        name="CharacterDesignerAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Using {story_spec_json}, {plot_outline_json}, and {world_json},
        design 3â€“5 primary characters with goals, flaws, backstories.
        Output JSON list.
        """),
        output_key="characters_json",
    )

    scene_writer_agent = LlmAgent(
        name="SceneDraftingAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Write the story as 6â€“10 scenes using:
        {story_spec_json}, {plot_outline_json}, {world_json}, {characters_json}.
        Output JSON: { "scenes": [ ... ] }
        """),
        output_key="story_draft_json",
    )

    return [
        story_requirements_agent,
        plot_agent,
        world_agent,
        character_agent,
        scene_writer_agent,
    ]



def create_critique_agents():
    story_critic = LlmAgent(
        name="StoryCriticAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Critique story pacing, logic, and clarity using {story_draft_json}.
        Output JSON: { "issues": "...", "suggestions": "..." }
        """),
        output_key="story_critique_json",
    )

    character_critic = LlmAgent(
        name="CharacterEmotionCriticAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Evaluate emotional depth and consistency using {story_draft_json} and {characters_json}.
        Output JSON.
        """),
        output_key="character_critique_json",
    )

    continuity_critic = LlmAgent(
        name="ContinuityLogicCheckerAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Check world-rule continuity using {story_draft_json} and {world_json}.
        Output JSON.
        """),
        output_key="continuity_critique_json",
    )

    debate_moderator = LlmAgent(
        name="DebateModeratorAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Combine all critiques into a unified revision plan.
        Use:
        {story_critique_json}, {character_critique_json}, {continuity_critique_json}.
        Output JSON: { "summary_of_changes": "...", "scene_level_instructions": {...} }
        """),
        output_key="revision_plan_json",
    )

    story_rewriter = LlmAgent(
        name="StoryRewriterAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Apply {revision_plan_json} to {story_draft_json}, rewriting scenes as needed.
        Output JSON: { "scenes": [...] }
        """),
        output_key="final_story_json",
    )

    final_summary = LlmAgent(
        name="FinalStorySummaryAgent",
        model=BASE_MODEL,
        instruction=dedent("""
        Build a readable final story with:
        - overview
        - final scenes from {final_story_json}
        Output plain text.
        """),
    )

    return [
        story_critic,
        character_critic,
        continuity_critic,
        debate_moderator,
        story_rewriter,
        final_summary,
    ]



# Build sub-agent lists
story_agents = create_story_agents()
critique_and_final_agents = create_critique_agents()

# Put them in execution order
sub_agents = story_agents + critique_and_final_agents

workflow_agent = SequentialAgent(
    name="StoryDebateWorkflow",
    sub_agents=sub_agents,
    description=(
        "Workflow: story spec â†’ plot â†’ world â†’ characters â†’ scenes â†’ "
        "critiques â†’ debate â†’ final rewritten story in scenes."
    ),
)

runner = InMemoryRunner(agent=workflow_agent, app_name=APP_NAME)

async def create_session():
    return await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

# Use `await` 
session = await create_session()
print("Created session:", session.id)



async def run_story_debate_pipeline_with_events(user_prompt: str):
    """
    Runs the full story pipeline and returns:
      - final_output_text: what FinalStorySummaryAgent returns
      - events: list of ADK Event objects for visualization
    """
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=user_prompt)],
    )

    events = []
    final_output_text = None

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message,
    ):
        # Collect every event
        events.append(event)

        # Final answer (from FinalStorySummaryAgent)
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
            final_output_text = text.strip()

    return final_output_text, events




def summarize_events(events) -> List[Dict[str, Any]]:
    """
    Turn ADK Event objects into a list of dicts:
      - index
      - author
      - kind
      - text
    """
    summary = []

    for idx, event in enumerate(events):
        author = getattr(event, "author", "unknown")

        # Default classification
        kind = "Other"

        # Detect content
        content = getattr(event, "content", None)
        parts = content.parts if (content and content.parts) else []

        # Safely extract first text part
        text = ""
        if parts:
            first = parts[0]
            text = getattr(first, "text", "") or ""

        # Tool calls / responses
        try:
            calls = event.get_function_calls()
            responses = event.get_function_responses()
        except AttributeError:
            calls = []
            responses = []

        actions = getattr(event, "actions", None)
        has_state_delta = bool(getattr(actions, "state_delta", None)) if actions else False
        has_artifact_delta = bool(getattr(actions, "artifact_delta", None)) if actions else False

        if author == "user":
            kind = "User message"
        elif calls:
            kind = "Tool call request"
        elif responses:
            kind = "Tool result"
        elif text:
            if getattr(event, "partial", False):
                kind = "Streaming text chunk"
            else:
                kind = "Agent message"
        elif has_state_delta or has_artifact_delta:
            kind = "State/artifact update"
        else:
            kind = "Control / other"

        summary.append(
            {
                "step": idx,
                "author": author,
                "kind": kind,
                "text": text.strip(),
            }
        )

    return summary



def render_agent_flow_ui(event_summary):
    """
    Render a simple HTML UI inside the notebook showing:
      - step number
      - author (user/agent)
      - kind
      - message text
    """

    css = """
    <style>
    .flow-container {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.4;
        max-width: 900px;
        margin: 1rem 0 3rem 0;
    }
    .flow-item {
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        border: 1px solid #e0e0e0;
        background: #fafafa;
    }
    .flow-item.user {
        border-left: 4px solid #1e88e5;
        background: #e3f2fd;
    }
    .flow-item.agent {
        border-left: 4px solid #43a047;
        background: #e8f5e9;
    }
    .flow-header {
        font-size: 0.8rem;
        margin-bottom: 4px;
        color: #555;
    }
    .flow-step {
        font-weight: 600;
        margin-right: 8px;
    }
    .flow-author {
        font-weight: 600;
    }
    .flow-kind {
        font-style: italic;
        color: #777;
        margin-left: 4px;
    }
    .flow-text {
        white-space: pre-wrap;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    </style>
    """

    html_chunks = [css, '<div class="flow-container">']

    for item in event_summary:
        author = item["author"]
        kind = item["kind"]
        step = item["step"]
        text = item["text"]

        role_class = "user" if author == "user" else "agent"

        html_chunks.append(f'''
        <div class="flow-item {role_class}">
          <div class="flow-header">
            <span class="flow-step">Step {step}</span>
            <span class="flow-author">{html.escape(author)}</span>
            <span class="flow-kind">({html.escape(kind)})</span>
          </div>
          <div class="flow-text">{html.escape(text) or "<no text / state update>"}</div>
        </div>
        ''')

    html_chunks.append("</div>")

    display(HTML("".join(html_chunks))
)



example_request = """
Create a dark fantasy hero's journey about a cursed knight who must choose
between saving the kingdom and freeing themselves from an ancient blood curse.
Tone should be serious but end with a hint of hope. Target a medium-length story
with clear scenes and strong character emotions. Audience: young adults.
"""

final_text, events = await run_story_debate_pipeline_with_events(example_request)

print("=== FINAL STORY (SCENES & DESCRIPTION) ===\n")
print(final_text)
print("\n\n=== AGENT FLOW VISUALIZATION ===\n")

event_summary = summarize_events(events)
render_agent_flow_ui(event_summary)






