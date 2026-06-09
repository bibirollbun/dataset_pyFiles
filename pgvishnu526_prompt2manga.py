!pip install "protobuf==3.20.3" diffusers transformers accelerate safetensors fpdf2



import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

print("âœ… API Key successfully loaded into Environment.")


import json
import torch
import time
from fpdf import FPDF
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ============================================================
# 1. MODEL SETUP â€“ Stable Diffusion (Anything V5 for Manga/Anime)
# ============================================================

print("ğŸ”„ Loading Anime Model...")
try:
    # "Anything V5" is a Stable Diffusion variant tuned for anime/manga style artwork
    model_id = "stablediffusionapi/anything-v5"
    
    # Load the diffusion pipeline in half precision (float16) on GPU
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Use an efficient scheduler for faster and cleaner images
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Disable the safety checker to allow action scenes / intense visuals
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    print("âœ… Model Loaded & Safety Checker Disabled (Action Scenes Allowed).")
except Exception as e:
    # If model load fails, we keep pipe=None so later tools can fall back gracefully
    print(f"âš ï¸� Model Load Error: {e}")
    pipe = None


# ============================================================
# 2. GLOBAL PATHS â€“ Where we store images, script, and state
# ============================================================

WORK_DIR   = "/kaggle/working"
IMG_DIR    = os.path.join(WORK_DIR, "images")          # All generated panel images
SCRIPT_FILE = os.path.join(WORK_DIR, "manga_script.json")  # Script from the LLM/agent
STATE_FILE  = os.path.join(WORK_DIR, "manga_state.json")   # Script + image paths for PDF

os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================
# 3. HELPER â€“ Fix encoding issues for PDF text
# ============================================================

def sanitize_text(text):
    """
    Cleans Unicode punctuation (like smart quotes, em-dash, ellipsis)
    and converts text to a PDF-safe Latin-1 representation.
    """
    if not isinstance(text, str):
        return str(text)
    
    replacements = {
        "\u2026": "...",  # ellipsis
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2013": "-",    # en dash
        "\u2014": "-"     # em dash
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Ensure the final string is Latin-1 compatible for FPDF
    return text.encode('latin-1', 'replace').decode('latin-1')


# ============================================================
# 4. TOOL 1 â€“ Save Script from Agent
# ============================================================

def save_script_tool(json_input: str):
    """
    TOOL 1 â€“ Accepts a JSON string from the LLM/agent, 
    cleans it, parses it, and saves to SCRIPT_FILE.
    
    Input : json_input (string, may contain ```json ... ``` wrapper)
    Output: Status string ("SCRIPT_SAVED" or error message)
    """
    try:
        # Remove common Markdown fencing like ```json ... ```
        clean = json_input.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON into Python structure
        data = json.loads(clean)
        
        # Save pretty-printed JSON for easier debugging
        with open(SCRIPT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return "SCRIPT_SAVED"
    except Exception as e:
        return f"Save Error: {e}"


# ============================================================
# 5. TOOL 2 â€“ Generate Images from Script
# ============================================================

def generate_images_tool(_ignored_input: str):
    """
    TOOL 2 â€“ Reads the saved script and generates panel images.
    
    - Reads SCRIPT_FILE
    - For each scene:
        - Build a Stable Diffusion prompt from description + mood
        - Generate an image (if model is loaded)
        - Attach image path back to the scene
    - Saves merged result to STATE_FILE
    
    Input : (unused, required by tool interface)
    Output: Status string ("IMAGES_GENERATED" or error message)
    """
    try:
        if not os.path.exists(SCRIPT_FILE):
            return "Error: No script file."
        
        # Load script (could be dict or nested JSON string)
        with open(SCRIPT_FILE, 'r') as f:
            data = json.load(f)

        if isinstance(data, str):
            data = json.loads(data)
            
        scenes = data.get('scenes', [])
        updated_scenes = []
        
        print(f"ğŸ�¨ Generating {len(scenes)} Panels...")
        
        for i, scene in enumerate(scenes):
            # Ensure scene is always a dict for safety
            if isinstance(scene, str):
                scene = json.loads(scene)
                
            desc = scene.get('description', '')
            mood = scene.get('mood', '')
            img_path = os.path.join(IMG_DIR, f"scene_{i+1}.png")
            
            if pipe:
                # Positive prompt: emphasize manga style, clean art
                prompt = (
                    "masterpiece, best quality, manga style, monochrome, greyscale, lineart, "
                    f"{desc}, {mood}, intense action, highly detailed, 4k"
                )
                
                # Negative prompt: avoid unwanted artifacts
                negative = (
                    "color, 3d, realistic, blurry, messy, sketch, text, watermark, "
                    "bad anatomy, deformed"
                )
                
                # Use autocast for faster, memory-efficient generation
                with torch.autocast("cuda"):
                    image = pipe(
                        prompt,
                        negative_prompt=negative,
                        num_inference_steps=25,  # speed/quality trade-off
                        width=512,
                        height=768,
                        guidance_scale=8.0
                    ).images[0]
                
                image.save(img_path)
                scene['image_path'] = img_path
            else:
                # Fallback if model failed to load; keeps pipeline alive for testing
                scene['image_path'] = "dummy.png"
            
            updated_scenes.append(scene)
        
        # Save combined state (script + image paths)
        with open(STATE_FILE, 'w') as f:
            json.dump({"scenes": updated_scenes}, f, indent=2)
            
        return "IMAGES_GENERATED"
    except Exception as e:
        return f"Gen Error: {e}"


# ============================================================
# 6. TOOL 3 â€“ Create Manga-Style PDF from Images + Script
# ============================================================

def create_pdf_tool(_ignored_input: str):
    """
    TOOL 3 â€“ Uses STATE_FILE (scenes + image paths) to create a manga PDF.
    
    Layout:
    - A4 portrait
    - 2 scenes (panels) per page: top and bottom
    - Each panel: image + narration box + dialogue box
    
    Input : (unused, required by tool interface)
    Output: Status string with PDF path, or error message
    """
    try:
        if not os.path.exists(STATE_FILE):
            return "Error: No state file."
        
        # ---- 1. Load and robustly parse state JSON ----
        with open(STATE_FILE, 'r') as f:
            raw_data = f.read().strip()
        
        data = None
        data_str = raw_data
        
        # Try multiple times in case of nested JSON strings
        for _ in range(5):
            try:
                data = json.loads(data_str)
                if not isinstance(data, str):
                    break  # parsed into final structure (dict)
                data_str = data
            except:
                break
        
        scenes = data.get('scenes', []) if isinstance(data, dict) else []
        
        # ---- 2. Setup PDF (A4) ----
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=False)  # manual layout
        
        # Helper: draw narration/dialog boxes on top of the panel image
        def draw_text_box(pdf, text, x, y, w, h, is_dialogue=False):
            if not text:
                return
            
            text = sanitize_text(text)
            
            # Dialogue: white bubble, Narration: light grey box
            if is_dialogue:
                pdf.set_fill_color(255, 255, 255)    # white
            else:
                pdf.set_fill_color(240, 240, 240)    # light grey
                
            pdf.set_draw_color(0, 0, 0)   # black border
            pdf.set_line_width(0.3)
            
            # Draw the filled rectangle (box)
            pdf.rect(x, y, w, h, style='FD')
            
            # Text properties
            pdf.set_xy(x + 2, y + 2)
            pdf.set_text_color(0, 0, 0)
            font_size = 10 if is_dialogue else 9
            font_style = 'B' if is_dialogue else 'I'
            pdf.set_font("Arial", font_style, font_size)
            
            # Wrap text inside the box
            pdf.multi_cell(w - 4, 5, text, align='C' if is_dialogue else 'L')

        # ---- 3. Layout each scene as a panel ----
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue

            # New page for every 2 scenes
            if i % 2 == 0:
                pdf.add_page()
                
                # Optional title only on the first page
                if i == 0:
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "AI Generated Manga", ln=True, align='C')

            # Top or bottom half of the page
            base_y = 20 if (i == 0) else 10
            y = base_y if (i % 2 == 0) else 148  # 2 panels per page
            
            img_path = scene.get('image_path', '')
            x_img = 10
            w_img = 190
            h_img = 130
            
            # Draw panel image with a border
            if os.path.exists(img_path):
                pdf.image(img_path, x=x_img, y=y, w=w_img, h=h_img)
                pdf.set_draw_color(0, 0, 0)
                pdf.set_line_width(1.0)
                pdf.rect(x_img, y, w_img, h_img)

            # Short narration box (top-left)
            narrative = scene.get('description', '')[:100]
            if narrative:
                draw_text_box(
                    pdf, narrative,
                    x=x_img + 2, y=y + 2,
                    w=80, h=20,
                    is_dialogue=False
                )
            
            # Dialogue: can be list of lines or a single string
            dialogue = scene.get('dialogue')
            dialogue_text = ""
            
            if isinstance(dialogue, list):
                for d in dialogue:
                    line = d.get('line', '') if isinstance(d, dict) else str(d)
                    dialogue_text += line + " "
            elif isinstance(dialogue, str):
                dialogue_text = dialogue
            
            # Dialogue bubble (bottom-right)
            if len(dialogue_text) > 0:
                # Adjust height based on approximate length
                box_h = 20 + (len(dialogue_text) // 30) * 5
                draw_text_box(
                    pdf, dialogue_text,
                    x=x_img + 110,
                    y=y + h_img - box_h - 5,
                    w=75, h=box_h,
                    is_dialogue=True
                )
        
        # ---- 4. Save final PDF ----
        out_path = "/kaggle/working/Final_Manga_Comic.pdf"
        pdf.output(out_path)
        
        return f"PDF_CREATED: {out_path}"
        
    except Exception as e:
        return f"PDF Error: {e}"



!pip install -U google-genai google-adk



from google.adk.agents import LlmAgent,SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types


retry_config=types.HttpRetryOptions(
    attempts=5,  
    exp_base=7, 
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], 
)


config_model = Gemini(
    model="gemini-2.5-flash-lite", 
    retry_options=retry_config
)


# 1. CHARACTER AGENT

char_agent = LlmAgent(
    name ="character_agent",
    model = config_model,
    instruction="""
        Role: Lead Character Designer.
        Task: Define the main protagonist for a Manga.
        Output Requirement: strictly valid JSON with keys:
        - "name": Character name.
        - "type": EXACTLY one of: ["Person", "Animal", "Fruit", "Object"].
        - "appearance": A dense, comma-separated string of visual keywords matching the 'type'.
          (e.g., "cybernetic wolf, metallic fur, glowing red eyes, forest background").
        - "personality": Brief archetype.
        
        NO CHAT. ONLY JSON.
        """,
    output_key = "char_data"
    
)


# 2. STORY AGENT

story_agent = LlmAgent(
    name="story_agent",
    model=config_model,
    instruction="""
        Role: Manga Director.
        Input: {char_data}
        Task: Create a detailed **10-panel story sequence** (approx 5 pages).
        
        Structure the pacing carefully:
        - Scenes 1-2: Introduction (Establish the setting and the character).
        - Scenes 3-4: The Conflict (An enemy appears or a problem starts).
        - Scenes 5-8: The Action/Climax (Dynamic battles, chases, or intensity).
        - Scenes 9-10: Resolution (The aftermath and cool ending pose).

        Output Requirement: strictly valid JSON with key "scenes", containing a list of 10 objects. Each object must have:
        - "scene_index": The number (1-10).
        - "description": Detailed visual instructions for the artist. **Focus on ACTION and CAMERA ANGLES** (e.g., "Low angle shot," "Close up on eyes," "Wide shot of explosion").
        - "mood": The atmosphere (e.g., "Tense", "Victory", "Dark").
        
        **CRITICAL SAFETY**: Describe action with 'energy', 'impact', 'motion blur'. Do NOT use 'blood' or 'gore'.
        NO CHAT. ONLY JSON.
        """,
    output_key = "story_data"
)


# 3. SCRIPT AGENT( Uses Tool)

script_agent = LlmAgent(
    name="dialogue_agent",
    model=config_model,
    instruction="""
        Role: Manga Script Writer.
        Input: {story_data}
        Task: 
        1. Review the full 10-scene sequence to understand the flow.
        2. Add 'dialogue' to EACH scene. 
           - Early scenes: Set the mystery.
           - Middle scenes: Short, punchy shouts or sound effects (SFX).
           - Final scenes: A cool one-liner.
        3. CALL `save_script_tool` with the full JSON.
        4. Output the tool result.
        
        Constraint: Keep dialogue under 20 words per bubble.
        NO CHAT.
        """,
    output_key = "save_status",
    tools=[save_script_tool] 
)


# 4. ILLUSTRATOR AGENT (Uses Tool)

illustrator_agent = LlmAgent(
    name="illustrator_agent",
    model=config_model,
    instruction="""
        Role: Render Engine Trigger.
        Task: The script is ready. Initiate the visual pipeline.
        Action: CALL `generate_images_tool` with argument 'start'.
        Do not explain. JUST CALL THE TOOL.
        """,
    output_key = "illus_status",
    tools=[generate_images_tool] 
)


# 5. PUBLISHER AGENT (Uses Tool)

publisher_agent = LlmAgent(
    name="publisher_agent",
    model=config_model,
     instruction="""
        Role: Publisher.
        Task: CALL `create_pdf_tool` with argument 'start'.
        Output the filename.
        NO CHAT.
        """,
    description="Compiles the PDF.",
    tools=[create_pdf_tool] 
)


# ROOT AGENT (Sequential Pattern)

root_agent = SequentialAgent(
    name="root",
    sub_agents=[
        char_agent,   
        story_agent,      
        script_agent,     
        illustrator_agent,  
        publisher_agent     
    ],
    
    description="Robust Manga Pipeline"
)


from google.adk.runners import InMemoryRunner 

runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug(
    "A boy going to forest with his grandpa and cousin , they stay nigth there and suddenly a meteor falling on earth near the place they staying and boy run to say it and goes close to say that a watch cameout form the box  and pasted on his hand and suddenly went to alien transformation"
)





