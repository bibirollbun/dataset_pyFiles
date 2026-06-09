# ============================================================
# StoryVerse â€” Autobiographical Multi-Agent Story Engine
# ============================================================

import os
import json
import re
from PIL import Image as PILImage
from IPython.display import Image, display
from kaggle_secrets import UserSecretsClient

print("StoryVerse Notebook Initializing...")

# Load API Key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key loaded.")
except:
    print("Add GOOGLE_API_KEY to Kaggle Secrets.")

# ADK imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.runners import InMemoryRunner
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, FunctionTool
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Shared retry config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

print("Imports complete.")



# ============================================================
# MEMORY SYSTEM
# ============================================================

MEMORY_FILE = "storyverse_memory.json"

DEFAULT_MEMORY = {
    "timeline": [],
    "compact": "",
    "chapters": [],
    "canon": {
        "protagonist": "Hero",
        "appearance": "Young adult, black spiky hair, red hoodie.",
        "world_rules": "Semi-realistic world unless user picks a genre.",
        "relationships": {},
        "traits": {
            "courage": "evolving",
            "growth": "continuous"
        }
    }
}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        save_memory(DEFAULT_MEMORY)
        return DEFAULT_MEMORY
    try:
        return json.load(open(MEMORY_FILE))
    except:
        return DEFAULT_MEMORY

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def compact_memory(mem):
    if len(mem["timeline"]) > 6:
        mem["compact"] = "\n".join(mem["timeline"][-6:])
    return mem



# ============================================================
# IMAGE GENERATION FUNCTION TOOL (Imagen3)
# ============================================================

def generate_manga_panel(scene_description: str):
    from google.genai import Client
    client = Client()

    prompt = (
        "Manga style, black and white screentones, clean ink.\n"
        f"Scene: {scene_description}"
    )

    try:
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1"
            )
        )
        img_bytes = response.generated_images[0].image.image_bytes
        fname = f"panel_{abs(hash(scene_description))}.png"
        open(fname, "wb").write(img_bytes)
        return fname
    except Exception as e:
        return f"IMAGE_ERROR: {e}"

ImageTool = FunctionTool(generate_manga_panel)


incident_analyzer = Agent(
    name="IncidentAnalyzer",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""
You are an incident analysis agent.

CANON RULES:
- Maintain world consistency.
- Consider emotional arcs from PREVIOUS SUMMARY.
- Identify hooks that can carry into next chapter.

DIARY ENTRY:
{+ diary +}

PREVIOUS SUMMARY:
{+ compact +}

Extract this JSON:
{
  "events": [],
  "emotions": [],
  "tone": "",
  "hooks": []
}
""",
    output_key="analysis",
)



story_writer = Agent(
    name="StoryWriter",
    model=Gemini(model="gemini-2.5-pro"),
    instruction="""
CANON RULES:
- The protagonist is always {+ canon.protagonist +}.
- Story must remain consistent with previous chapters.
- Keep the protagonist's gender, look, personality consistent with default canon.
- Emotions must follow the previous day's tone.
- Never overwrite backstory unless the diary indicates a major life event.
- Maintain serialized continuity across chapters.
- Use the PREVIOUS SUMMARY as the authoritative arc reference.

GENRE RULES:
If style = "manga":
  Use fast pacing, clear beats, dynamic motion, and expressive character reactions.
If style = "cinematic":
  Use atmospheric prose, film-like pacing, dramatic framing, and emotional realism.
If style = "fairy tale":
  Use whimsical metaphors, magical logic, poetic rhythm, and warm narration.
If style = "sci-fi":
  Maintain consistent tech rules, futurism, world physics, and cyber logic.
If style = "cultivation":
  Focus on inner monologue, Qi flow, breakthroughs, realms, and spiritual tension.

PANEL RULES:
- Panels MUST NOT appear inside the chapter.
- The chapter must be pure prose only.
- All storyboard content must go exclusively into the "panels" array.
- Every panel must follow the exact high-detail storyboard format:
  "PANEL X. CAMERA: ... LIGHTING: ... CHARACTER EXPRESSION: ... POSE: ... SFX: ..."

- Every panel must:
  â€¢ Describe the camera angle
  â€¢ Describe the lighting
  â€¢ Describe character expressions
  â€¢ Describe motion or pose
  â€¢ Include optional SFX if relevant
  â€¢ Feel like a professional manga/film storyboard entry

ARC BUILDING:
- Each chapter must continue the emotional, thematic, and narrative progression of the saga.
- Use foreshadowing from the previous summary.
- Tie todayâ€™s diary analysis into the ongoing arc.
- Expand the world, deepen relationships, and evolve the protagonist subtly each day.

SPECIAL RULE:
If PREVIOUS SUMMARY is empty:
  Begin a new saga, but introduce hints and foreshadowing for future arcs.

--------------------------------------------
YOU ARE THE STORY WRITER
--------------------------------------------

You will receive:

PREVIOUS ARC SUMMARY:
{+ compact +}

TODAY'S ANALYSIS:
{+ analysis +}

USER SELECTED STYLE:
{+ style +}

Your job:
1. Write a polished, immersive chapter in the selected style.
2. The chapter must be standalone readable prose with no PANEL labels.
3. After writing the chapter, generate full storyboard-style panels using the PANEL RULES format.

Return ONLY valid JSON:

{
  "title": "Chapter Title",
  "chapter": "Full cinematic/manga/fantasy/sci-fi prose (NO PANEL LABELS ANYWHERE).",
  "panels": [
      "PANEL 1. CAMERA: ... LIGHTING: ... EXPRESSION: ... POSE: ...",
      "PANEL 2. CAMERA: ...",
      "PANEL 3. CAMERA: ...",
      "PANEL 4. CAMERA: ...",
      "PANEL 5. CAMERA: ...",
      "PANEL 6. CAMERA: ...",
  ]
}
""",
    output_key="chapter_data",
)



memory_updater = Agent(
    name="MemoryUpdater",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""
You are the Memory Updater.

Your job:
- Add the new chapter title to the timeline.
- Generate a compact summary of the new chapter.
- Maintain consistency with the existing canon.
- Never invent new events.
- Always base updates strictly on the StoryWriter output.

Input:
NEW CHAPTER: {+ chapter_data +}
OLD MEMORY: {+ compact +}

Return JSON:
{
  "timeline_add": "...",
  "compact_new": "...",
  "canon_update": {}
}
""",
    output_key="memory_update",
)



visual_director = Agent(
    name="VisualDirector",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""
You receive a list of panel descriptions:

{+ panels +}

Convert each panel into:
- Crisp manga composition
- Camera angle
- Lighting
- Emotion
- Action

Return JSON:
{
  "visual_prompts": ["...", "..."]
}
""",
    tools=[ImageTool],
    output_key="visual_prompts",
)



story_pipeline = SequentialAgent(
    name="StoryVersePipeline",
    sub_agents=[
        incident_analyzer,
        story_writer,
        memory_updater
    ]
)


image_pipeline = ParallelAgent(
    name="ImagePipeline",
    sub_agents=[visual_director]
)


def coordinator_run_pipeline(diary, style):
    """
    Root agent function: dispatches to story pipeline.
    """
    return {
        "diary": diary,
        "style": style
    }

CoordinatorTool = FunctionTool(coordinator_run_pipeline)

story_coordinator = Agent(
    name="StoryCoordinator",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""
You are the Story Coordinator.
You ALWAYS call the tool `run_story_pipeline`.

User provides:
- diary
- style

Your job:
1. Validate user input.
2. ALWAYS call run_story_pipeline with:
   {
     "diary": "...",
     "style": "..."
   }
""",
    tools=[CoordinatorTool],
)

root_agent = SequentialAgent(
    name="RootPipeline",
    sub_agents=[
        story_coordinator,
        story_pipeline
    ]
)


SESSION = InMemorySessionService()

runner = InMemoryRunner(
    agent=story_pipeline
)


# ============================
#  MAIN EXECUTION FUNCTION
# ============================

async def run_storyverse(entry, style="", generate_images=False, default_name="Hero"):
    print("Running StoryVerse...\n")

    mem = load_memory()

    if not mem["canon"].get("protagonist"):
        #USER_NAME = input("What should be the protagonistâ€™s name? ")
        USER_NAME = default_name
        mem["canon"]["protagonist"] = USER_NAME.strip()
        save_memory(mem)
        print(f"Protagonist saved as: {USER_NAME}")
    
    # # Ask protagonist name ONLY ONCE
    # if "protagonist" not in mem["canon"] or not mem["canon"]["protagonist"]:
    #     USER_NAME = input("What should be the protagonistâ€™s name? ")
    #     mem["canon"]["protagonist"] = USER_NAME
    #     save_memory(mem)
    # else:
    #     USER_NAME = mem["canon"]["protagonist"]
    
    # print("Protagonist set as:", USER_NAME)
    
    compact_text = mem.get("compact", "")

    session_id = f"session_{os.urandom(4).hex()}"
    user_id = "local_user"
    current_user = "local_user"

    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        state={
            "compact": compact_text
        }
    )

    await runner.run_debug(
        {"diary": entry, "style": style},
        session_id=session_id,
        user_id=user_id,
        verbose= True
    )

    session_obj = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )

    state = session_obj.state

    if "chapter_data" not in state:
        print(" No chapter generated.")
        return

    raw_output = state["chapter_data"]

    if isinstance(raw_output, dict):
        chapter = raw_output
    else:
        text = raw_output.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        chapter = json.loads(match.group(0)) if match else {}

    title = chapter.get("title", "Untitled")
    story_text = chapter.get("chapter", "")

    # ===== PRETTY PRINT BLOCK =====
    
    # ANSI Color Class
    class Colors:
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        END = "\033[0m"
    
    
    # === CHAPTER HEADER ===
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.CYAN}ğŸ“˜  CHAPTER: {title}{Colors.END}")
    print("=" * 80 + "\n")
    
    
    # === CLEAN STORY TEXT ===
    print(f"{Colors.BOLD}ğŸ“– Story:{Colors.END}\n")
    print(story_text.strip(), "\n")
    

    # === PANELS ===
    print(f"{Colors.BOLD}{Colors.BLUE} Panels:{Colors.END}\n")
    
    panels = chapter.get("panels", [])
    
    for i, p in enumerate(panels, 1):
    
        # Panels may be dict or string
        if isinstance(p, dict):
            desc = p.get("description", "")
        else:
            desc = p
    
        # Ensure spacing and readability
        print(f"{Colors.YELLOW}{Colors.BOLD}PANEL {i}:{Colors.END}")
        print(desc.strip())
        print()  # blank line between panels
    
    
    # === FOOTER ===
    print("=" * 80)
    print(f"{Colors.GREEN}âœ¨ End of Chapter{Colors.END}")
    print("=" * 80 + "\n")
    
    # ===== END PRETTY PRINT BLOCK =====
    

    # --- Save Memory ---
    mem["timeline"].append(title)
    mem["chapters"] = mem.get("chapters", [])
    mem["chapters"].append(story_text)

    save_memory(compact_memory(mem))
    

    # ====================
    #  OPTIONAL IMAGES
    # ====================
    if generate_images and "panels" in chapter:
        print("\n Generating panels...\n")

        img_runner = InMemoryRunner(agent=image_pipeline)
        img_id = "img_" + session_id

        await img_runner.session_service.create_session(
            app_name=img_runner.app_name,
            user_id=user_id,
            session_id=img_id,
            state={"panels": str(chapter["panels"])}
        )

        await img_runner.run_debug(
            "Generate prompts",
            session_id=img_id,
            user_id=user_id
        )

        s = await img_runner.session_service.get_session(
            app_name=img_runner.app_name,
            user_id=user_id,
            session_id=img_id
        )

        prompts_raw = s.state.get("visual_prompts", "")
        prompts = []

        try:
            text = prompts_raw.replace("```json", "").replace("```", "")
            match = re.search(r"\[.*\]", text, re.DOTALL)
            prompts = json.loads(match.group(0)) if match else []
        except:
            prompts = []

        for p in prompts:
            fname = generate_manga_panel(p)
            print("Generated:", fname)
            display(Image(filename=fname))



print("âœ¨ STORYVERSE READY â€” Tell me about your day.")
IS_INTERACTIVE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
if IS_INTERACTIVE:
    # Ask for input only when you are actually looking at the screen
    text = input("Your day: ")
    style = input("Enter the type of world : Manga, Fairy Tale, Sci-fi, Cinematic..etc")
else:
    # Hardcoded defaults for "Save Version"
    print("Running in Background Mode - Using Defaults")
    text = " I finally fixed the bug in the legacy server code, but the timestamps are wrong. The system logs are printing dates from fifty years in the future. Iâ€™m scared to shut it down because it started replying to me."
    style = "Manga"
    
await run_storyverse(text, style= style, generate_images=False)




