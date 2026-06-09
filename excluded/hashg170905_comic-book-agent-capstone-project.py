import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

IMAGE_MODEL = "gemini-2.5-flash-image"
TEXT_MODEL = "gemini-2.5-flash"


import os
import logging
from typing import Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, field_validator

from google import genai
from google.genai import types

from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner, InMemoryRunner


print("âœ… ADK components imported successfully.")


UNSAFE_WORDS = {
    "scary", "frightening", "weapon", "gun", "knife", "violence", "fight",
    "hurt", "pain", "blood", "danger", "evil", "monster", "ghost", "nightmare",
    "death", "kill", "hate"
}


def validate_content_safety(
    callback_context: CallbackContext
) -> Optional[LlmResponse]:
    """
    Basic safety check for kid-friendly content.

    Validates that generated content doesn't contain inappropriate themes.
    """
    # Get the generated content
    state = callback_context.state.to_dict()

    # Check story content
    if "story" in state:
        story_data = state["story"]
        story_text = story_data.get("story_text", "")
        moral_lesson = story_data.get("moral_lesson", "")

        combined_text = f"{story_text} {moral_lesson}".lower()

        found_unsafe = [word for word in UNSAFE_WORDS if word in combined_text]

        if found_unsafe:
            logger.warning(f"[Safety] Found unsafe words: {found_unsafe}")
            # For demo purposes, just log a warning but allow continuation
            # In production, you might want to block and regenerate

    # Check comic panel content
    if "comic" in state:
        comic_data = state["comic"]
        panels = comic_data.get("panels", [])

        for panel in panels:
            panel_desc = panel.get("description", "").lower()
            dialogue = panel.get("dialogue", "").lower()
            combined = f"{panel_desc} {dialogue}"

            found_unsafe = [word for word in UNSAFE_WORDS if word in combined]

            if found_unsafe:
                logger.warning(
                    f"[Safety] Panel {panel.get('panel_number')} has unsafe words: {found_unsafe}"
                )

    # Allow continuation - just log warnings for demo
    return None


# Lazily created global client
_client = None


def _get_client():
    """Create the Gemini client using the new streamlined API."""
    global _client
    if _client is None:
        # Expects GOOGLE_API_KEY in environment variables
        _client = genai.Client()
    return _client


def generate_storyboard_image(
    panel_description: str,
    panel_number: int,
    character_descriptions: str,
    tool_context: ToolContext
) -> list[str]:
    """
    Generate a storyboard illustration using Gemini 2.5 Flash Image.
    Uses text-only input and produces a single image via generate_content.
    """

    try:
        # Session for directory naming
        session_id = tool_context._invocation_context.session.id
        output_dir = f"./output/{session_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Build prompt
        # Note: Since we aren't using a config object for aspect ratio in this specific
        # snippet pattern, we explicitly request the aspect ratio in the text prompt.
        full_prompt = f"""
Create a colorful, kid-friendly illustration in a vibrant children's book in comic style. 
The panel contains a speech bubble with the text.
Format: Wide 16:9 aspect ratio.

Scene:
{panel_description}

Characters (keep appearance consistent):
{character_descriptions}

Art direction:
- Bright colors, friendly expressions
- Soft rounded shapes
- Positive and cheerful tone (ages 3â€“10)

Avoid:
scary, dark, violent, frightening, sad, crying, weapons, monsters, or anything inappropriate.
"""

        logger.info(f"[Image Tool] Generating image for panel {panel_number}")

        client = _get_client()

        # 1. Use generate_content (not generate_images)
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=full_prompt,
            # Config is optional here. If the model supports specific image params
            # via config, they can be added, but we stick to your snippet's simplicity.
        )

        image_paths = []

        # 2. Iterate through parts to find inline_data (images)
        if response.parts:
            for idx, part in enumerate(response.parts):

                # Check if this part is an image
                if part.inline_data is not None:
                    # 3. Use the helper method .as_image() (returns PIL Image)
                    image = part.as_image()

                    filename = f"storyboard_panel_{panel_number}_{idx}.png"
                    local_path = os.path.join(output_dir, filename)

                    image.save(local_path)

                    abs_path = os.path.abspath(local_path)
                    image_paths.append(abs_path)

                    logger.info(f"[Image Tool] Saved image to {abs_path}")

                elif part.text is not None:
                    # Sometimes the model might return text alongside the image
                    logger.debug(f"[Image Tool] Model returned text: {part.text[:100]}...")

        if not image_paths:
            logger.warning(f"[Image Tool] No images generated for panel {panel_number}")

        return image_paths

    except Exception as e:
        logger.error(f"[Image Tool] Error generating image for panel {panel_number}: {e}")
        return []




class Character(BaseModel):
    """A character in the story with visual description."""

    name: str = Field(description="Character name")
    role: Literal["hero", "friend", "helper", "guide"] = Field(
        description="Character's role in the story"
    )
    description: str = Field(
        description="Detailed visual description for image generation (appearance, clothing, colors)"
    )
    personality: str = Field(
        description="Key personality traits in one sentence"
    )


class StoryWithCharacters(BaseModel):
    """Complete story with characters - simplified for competition demo."""

    title: str = Field(description="Catchy movie title")

    age_range: Literal["3-5", "5-7", "7-10"] = Field(
        default="5-7",
        description="Target age range"
    )

    moral_lesson: str = Field(
        description="The educational message or value taught"
    )

    story_text: str = Field(
        description="Complete story narrative (3-5 paragraphs, kid-friendly language)",
        min_length=200
    )

    characters: list[Character] = Field(
        description="Main characters (1-4 characters)",
        min_length=1,
        max_length=4
    )

    setting: str = Field(
        description="Where the story takes place (for visual consistency)"
    )

    # Pre-validate and trim story_text to satisfy downstream ADK runtime limit (1000 chars)
    @field_validator("story_text", mode="before")
    @classmethod
    def _trim_story_text(cls, v):
        if isinstance(v, str):
            return v[:1000]
        return v

class Panel(BaseModel):
    """A single panel in the comic strip."""

    panel_number: int = Field(ge=1, le=6, description="Panel number (1-6)")

    description: str = Field(
        description="What happens in this panel (visual action and composition)"
    )

    characters: list[str] = Field(
        description="Character names appearing in this panel"
    )

    dialogue: str = Field(
        default="",
        description="Speech bubbles, thought bubbles, or narration text"
    )

    mood: Literal["happy", "excited", "curious", "gentle", "peaceful", "joyful", "surprised", "thoughtful"] = Field(
        description="Emotional tone of the panel"
    )

    panel_layout: Literal["full-width", "half", "third"] = Field(
        default="half",
        description="Panel size in the comic layout"
    )

class PanelImage(BaseModel):
    """Image asset for a single comic panel."""

    panel_number: int = Field(description="Panel number")
    image_path: str = Field(description="Local path to panel image")
    dialogue: str = Field(default="", description="Text to overlay on panel")


class ComicAssets(BaseModel):
    """All generated images for the comic strip."""

    panels: list[PanelImage] = Field(
        description="Images for each panel"
    )

    output_directory: str = Field(
        description="Local directory where all panel images are saved"
    )

class ComicStrip(BaseModel):
    """Complete plan for all panels in the comic strip."""

    panels: list[Panel] = Field(
        description="4-6 panels that tell the complete story",
        min_length=4,
        max_length=6
    )

    art_style: Literal["cartoon", "manga", "children-book", "colorful-comic"] = Field(
        default="colorful-comic",
        description="Overall visual style of the comic"
    )


print("âœ… Models created.")


STORY_CREATOR_PROMPT = """You are a creative children's story writer and character designer.

Your task is to create an engaging, educational short story for children WITH detailed character descriptions.

Guidelines:
1. **Story Requirements:**
   - Create a complete story in 3-5 paragraphs
   - Include a clear moral lesson or positive value
   - Use age-appropriate language (simple words, short sentences)
   - Make it engaging and fun for kids ages 3-10
   - Set the story in a specific, visually interesting location

2. **Character Requirements:**
   - Create 1-4 memorable characters
   - Provide DETAILED visual descriptions for each character (colors, clothing, physical features)
   - Give each character a clear role (hero, friend, helper, guide)
   - Make characters diverse and inclusive

3. **Content Safety:**
   - NO scary, violent, or frightening content
   - Use positive, uplifting themes
   - Focus on: friendship, kindness, sharing, courage, helping others, cooperation
   - Keep everything cheerful and age-appropriate

4. **Visual Consistency:**
   - Describe characters in detail so they can be visualized consistently
   - Describe the setting clearly for visual reference
   - Use bright, colorful imagery

Remember: This story will be turned into an animated movie with images and videos!
"""


logger = logging.getLogger(__name__)

try:
    story_creator_agent = Agent(
        model="gemini-2.5-flash",
        name="story_creator",
        instruction=STORY_CREATOR_PROMPT,
        output_schema=StoryWithCharacters,
        output_key="story",
        after_agent_callback=validate_content_safety,
    )
    logger.info("âœ… Story Creator agent created successfully")
except Exception as e:
    logger.error(f"â�Œ Failed to create Story Creator agent: {e}")
    story_creator_agent = None

print("âœ… story_agent created.")


PANEL_PLANNER_PROMPT = """You are an expert at creating engaging comic strips for children.

Your task is to convert the story into 4-6 comic panels.

Guidelines:
1. **Panel Breakdown:**
   - Create 4-6 panels that tell the complete story
   - Each panel should be a distinct moment/beat
   - Think about visual storytelling and pacing
   - Include which characters appear in each panel

2. **Visual Composition:**
   - Describe what's happening visually in each panel
   - Think about character positions, expressions, and actions
   - Consider background elements and setting
   - Make each panel visually interesting and clear

3. **Dialogue/Text:**
   - Add simple, kid-friendly dialogue or narration
   - Keep text short and punchy (comic style)
   - Use speech bubbles, thought bubbles, or captions
   - Match the character's personality

4. **Mood & Emotion:**
   - Assign an emotional tone to each panel
   - Show emotions through character expressions and body language
   - Keep moods positive and appropriate for children

5. **Comic Flow:**
   - Ensure panels flow logically from one to the next
   - Include setup, development, and satisfying conclusion
   - Build to the moral lesson/happy ending
   - Make it easy to read left-to-right, top-to-bottom

6. **Panel Layout:**
   - Vary panel sizes for visual interest
   - full-width for important moments
   - half or third for regular panels
   - Create dynamic, engaging layouts

Remember: This will be a colorful, kid-friendly comic strip with vibrant illustrations!
"""

try:
    panel_planner_agent = Agent(
        model="gemini-2.5-flash",
        name="panel_planner",
        instruction=PANEL_PLANNER_PROMPT,
        output_schema=ComicStrip,
        output_key="comic",
        after_agent_callback=validate_content_safety,
    )
    logger.info("âœ… Panel Planner agent created successfully")
except Exception as e:
    logger.error(f"â�Œ Failed to create Panel Planner agent: {e}")
    panel_planner_agent = None

print("âœ… panel_planner_agent created.")


PANEL_GENERATOR_PROMPT = """You are a comic panel illustrator for children's content.

Your task is to generate colorful comic panel images for each panel in the comic strip.

Guidelines:
1. **For Each Panel:**
   - Call `generate_storyboard_image` for EVERY panel
   - Use the panel description for the image prompt
   - Include character descriptions for visual consistency
   - Add comic-style text and speech bubbles

2. **Tool Usage:**
   - Call `generate_storyboard_image` once per panel
   - Pass the panel_number correctly (starts at 1)
   - Include complete character descriptions
   - Reference the setting and mood

3. **Visual Consistency:**
   - Maintain consistent character appearance across all panels
   - Keep the art style uniform (colorful comic book style)
   - Use the same setting/background elements where appropriate
   - Ensure characters look the same in every panel

4. **Comic Style:**
   - Bright, vibrant colors
   - Clear character expressions
   - Kid-friendly, welcoming aesthetic
   - Dynamic compositions that tell the story visually
   - Add speech bubbles and text where appropriate

5. **Quality:**
   - Generate ALL panels (4-6 panels total)
   - Ensure each image matches its panel description
   - Maintain high visual quality throughout

You must call the tool for all panels to complete the comic strip!
"""

try:
    panel_generator_agent = Agent(
        model="gemini-2.5-flash",
        name="panel_generator",
        instruction=PANEL_GENERATOR_PROMPT,
        tools=[generate_storyboard_image],
        output_schema=ComicAssets,
        output_key="panels",
    )
    logger.info("âœ… Panel Generator agent created successfully")
except Exception as e:
    logger.error(f"â�Œ Failed to create Panel Generator agent: {e}")
    panel_generator_agent = None

print("âœ… panel_generator_agent created.")


# Validate all sub-agents
if not all([story_creator_agent, panel_planner_agent, panel_generator_agent]):
    raise RuntimeError("One or more sub-agents failed to initialize")

# Create root sequential agent

root_agent = SequentialAgent(
    name="comic_strips",
    description="Create kid-friendly comic strips from text prompts",
    sub_agents=[
        story_creator_agent,
        panel_planner_agent,
        panel_generator_agent,
    ],
)
print("âœ… root_agent created.")


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("Create a comic about a penguin learning to share")

