# ğŸ“º Watch the Project Demo (Official Video)
from IPython.display import YouTubeVideo, display, HTML

print("ğŸ�¬ Loading Project Demo Video...")

# Setting up the video player
video_id = "YRi4qMBVauo"
display(YouTubeVideo(video_id, width=800, height=450))

print("ğŸ‘† Watch how Creator's Co-Pilot works in 3 minutes!")


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%capture
# Cell 1: Install dependencies (Wikipedia Edition)

# Install Gemini
!pip install -q google-generativeai

# Install Wikipedia (The most reliable research tool)
!pip install -q wikipedia

print("Dependencies installed successfully.")


# Cell 2: Setup & SMARTER Wikipedia Tool

import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
import wikipedia

# --- 1. Configuration ---
print("System: Initializing configuration...")
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    print("System: SUCCESS - AI Model ready.")
except Exception as e:
    print(f"System Error: API Key missing. {e}")

# --- 2. SMART Wikipedia Tool Definition ---

def research_tool(query: str) -> str:
    """
    Searches Wikipedia smartly.
    First searches for the best matching page title, then fetches the summary.
    """
    try:
        print(f"ğŸ”� Tool: Searching for related pages to '{query}'...")
        
        # Step 1: Search for page titles
        search_results = wikipedia.search(query)
        
        if not search_results:
            return "System: No Wikipedia pages found."
            
        # Step 2: Pick the best result (the first one)
        best_page = search_results[0]
        print(f"ğŸ‘‰ Tool: Found best match -> '{best_page}'. Fetching data...")
        
        # Step 3: Get the summary of that specific page
        summary = wikipedia.summary(best_page, sentences=15, auto_suggest=False)
        
        return f"Wikipedia Summary for '{best_page}':\n{summary}"

    except Exception as e:
        return f"System Error: {e}"

# --- 3. Test ---
print("\nSystem: Testing Smart Tool...")
# à¤…à¤¬ à¤¯à¤¹ 'Elon Musk' à¤•à¥‹ à¤–à¥‹à¤œà¤•à¤° à¤¸à¤¹à¥€ à¤ªà¥‡à¤œ à¤²à¤¾à¤�à¤—à¤¾
print(research_tool("Elon Musk simulation"))


# Cell 3: Define Agent 1 (The Researcher) - Wikipedia Version

RESEARCHER_PROMPT = """
You are an expert YouTube Researcher. Your goal is to find high-engagement facts.

You will be provided with a Wikipedia summary of a topic.
Your task is to extract the most interesting facts for a video.

GUIDELINES:
1. Format: Return a bulleted list of 5-7 key facts.
2. Focus: Look for controversial, surprising, or major achievements.
"""

def run_research_agent(topic: str) -> str:
    print(f"Agent 1 (Researcher): Starting research on '{topic}'...")
    
    # Step 1: Get Data from Wikipedia
    raw_data = research_tool(topic)
    
    # Step 2: Create Prompt
    final_prompt = f"""
    {RESEARCHER_PROMPT}

    --- WIKIPEDIA DATA ---
    {raw_data}
    --- END DATA ---

    Topic: {topic}
    Extract the key facts.
    """
    
    # Step 3: Gemini Analysis
    try:
        response = model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return f"Agent Error: {e}"

print("System: Agent 1 (Wikipedia Version) is ready.")


# Cell 4: System Integration Test - Agent 1 Execution

# Define a test topic relevant to the YouTube niche
TEST_TOPIC = "Elon Musk simulation theory arguments facts"

print(f"System: Initiating workflow for topic: '{TEST_TOPIC}'")
print("-" * 60)

try:
    # Execute the Research Agent
    # This triggers the Search Tool -> LLM processing pipeline
    research_output = run_research_agent(TEST_TOPIC)
    
    # Display the Final Output
    print("\n--- ğŸ“„ AGENT 1 OUTPUT (Research Data) ---")
    print(research_output)
    print("-" * 60)
    
    print("System: Workflow completed successfully.")

except Exception as e:
    print(f"System Error: Workflow execution failed. Details: {e}")


# Cell 5: Define Agent 2 (The Script Writer)

# --- System Instructions ---
SCRIPT_WRITER_PROMPT = """
You are a professional YouTube Script Writer (like for MrBeast or Veritasium).
Your goal is to turn raw facts into a viral, high-retention video script.

INPUT: A list of research facts.
OUTPUT: A complete video script (Title, Hook, Intro, Body, Outro).

GUIDELINES:
1. Tone: Energetic, curious, and easy to understand.
2. Structure:
   - **HOOK (0-15s):** Grab attention immediately.
   - **INTRO:** Briefly explain what we are covering.
   - **BODY:** Present the facts in a storytelling format (not just a list).
   - **OUTRO:** Call to Action (Subscribe).
3. Formatting: Use [Visual Cues] in brackets to suggest what should be on screen.
"""

def run_script_agent(research_data: str) -> str:
    """
    Takes research data and generates a YouTube script.
    """
    print(f"Agent 2 (Writer): Reading research data...")
    
    # Create the Prompt
    # We chain the previous agent's output into this agent's input
    final_prompt = f"""
    {SCRIPT_WRITER_PROMPT}

    --- RESEARCH DATA ---
    {research_data}
    --- END DATA ---

    Write a script based on this data.
    """
    
    # Call Gemini
    print("Agent 2: Writing script with Gemini...")
    try:
        response = model.generate_content(final_prompt)
        print("Agent 2: Script complete.")
        return response.text
    except Exception as e:
        return f"Agent Error: {e}"

print("System: Agent 2 (Script Writer) module loaded successfully.")


# Cell 6: Integration Test - Agent 1 -> Agent 2

print("--- ğŸš€ EXECUTING PIPELINE: Research -> Script ---")

# We use the 'research_output' variable from the previous step (Cell 4)
# This is how we create a "Chain" of agents.

if 'research_output' in locals() and research_output:
    script_output = run_script_agent(research_output)
    
    print("\n--- ğŸ�¬ AGENT 2 OUTPUT (The Script) ---")
    print(script_output)
else:
    print("Error: No research data found. Please run Cell 4 first.")


# Cell 7: Define Agent 3 (The SEO Specialist)

# --- System Instructions ---
SEO_EXPERT_PROMPT = """
You are a YouTube SEO Expert (Search Engine Optimization).
Your goal is to maximize the Views (CTR) and Reach of a video based on its script.

INPUT: A full video script.
OUTPUT: A metadata package optimized for the YouTube Algorithm.

GUIDELINES:
1. Titles: Create 3 options (1. Curious/Clickbaity, 2. Direct/Searchable, 3. Story-driven).
2. Description: Write a compelling first paragraph + a bulleted list of what's inside.
3. Tags: Generate 15 high-volume keywords separated by commas.
4. Hashtags: Generate 3-5 relevant hashtags.
"""

def run_seo_agent(script_content: str) -> str:
    """
    Analyzes the script and generates titles, description, and tags.
    """
    print(f"Agent 3 (SEO): Analyzing script for keywords and hooks...")
    
    # Create the Prompt
    final_prompt = f"""
    {SEO_EXPERT_PROMPT}

    --- VIDEO SCRIPT ---
    {script_content}
    --- END SCRIPT ---

    Generate the SEO metadata package.
    """
    
    # Call Gemini
    print("Agent 3: Optimizing metadata with Gemini...")
    try:
        response = model.generate_content(final_prompt)
        print("Agent 3: SEO package ready.")
        return response.text
    except Exception as e:
        return f"Agent Error: {e}"

print("System: Agent 3 (SEO Specialist) module loaded successfully.")


# Cell 8: Integration Test - Agent 2 -> Agent 3

print("--- ğŸš€ EXECUTING PIPELINE: Script -> SEO ---")

# We use the 'script_output' variable from Cell 6
if 'script_output' in locals() and script_output:
    seo_output = run_seo_agent(script_output)
    
    print("\n--- ğŸ“ˆ AGENT 3 OUTPUT (SEO Metadata) ---")
    print(seo_output)
else:
    print("Error: No script found. Please run Cell 6 first.")


# Cell 9: Main Orchestration Workflow (The "Manager")

import time

def generate_full_video_package(topic: str):
    """
    The main entry point for Creator's Co-Pilot.
    This function orchestrates the entire multi-agent workflow:
    Research -> Scripting -> SEO.
    
    Args:
        topic (str): The video topic idea.
    """
    print(f"ğŸ�¬ STARTING PRODUCTION FOR: '{topic}'")
    print("=" * 60)
    
    start_time = time.time()

    # --- Phase 1: Research Agent ---
    print("\nğŸ“¡ [Step 1/3] Agent 1 (Researcher) is working...")
    research_data = run_research_agent(topic)
    print("âœ… Research collected.")

    # --- Phase 2: Script Writer Agent ---
    print("\nâœ�ï¸� [Step 2/3] Agent 2 (Script Writer) is working...")
    script_content = run_script_agent(research_data)
    print("âœ… Script drafted.")

    # --- Phase 3: SEO Agent ---
    print("\nğŸ“ˆ [Step 3/3] Agent 3 (SEO Specialist) is working...")
    seo_data = run_seo_agent(script_content)
    print("âœ… SEO metadata generated.")

    # --- Final Compilation ---
    duration = round(time.time() - start_time, 2)
    
    print("\n" + "=" * 60)
    print(f"âœ¨ PRODUCTION COMPLETE in {duration} seconds! âœ¨")
    print("=" * 60)
    
    # Display the Final Report nicely
    print("\n" + "#" * 20 + " PHASE 1: RESEARCH " + "#" * 20 + "\n")
    print(research_data)
    
    print("\n" + "#" * 20 + " PHASE 2: SCRIPT " + "#" * 20 + "\n")
    print(script_content)
    
    print("\n" + "#" * 20 + " PHASE 3: SEO " + "#" * 20 + "\n")
    print(seo_data)

print("System: Main Orchestrator loaded successfully.")


# Cell 10: Creator's Co-Pilot Interface (The Safe, Kaggle-Native Version)

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# --- Style & Layout ---
style = {'description_width': 'initial'}
header = HTML("<h2>ğŸ�¬ Creator's Co-Pilot: Control Panel</h2>")

# 1. Input Area
topic_input = widgets.Text(
    placeholder='e.g., The Philosophy of Batman',
    description='<b>Enter Video Topic:</b>',
    style=style,
    layout=widgets.Layout(width='80%')
)

# 2. Action Button
run_btn = widgets.Button(
    description='Generate Video Package ğŸš€',
    button_style='success', # Green button
    layout=widgets.Layout(width='30%', margin='10px 0px')
)

# 3. Output Tabs (Research, Script, SEO)
out_research = widgets.Output()
out_script = widgets.Output()
out_seo = widgets.Output()

tabs = widgets.Tab(children=[out_research, out_script, out_seo])
tabs.set_title(0, 'ğŸ•µï¸�â€�â™‚ï¸� Research')
tabs.set_title(1, 'âœ�ï¸� Script')
tabs.set_title(2, 'ğŸ“ˆ SEO Data')

# 4. The Logic
def on_click(b):
    topic = topic_input.value
    if not topic: return
    
    run_btn.description = "Processing... (Please Wait)"
    run_btn.disabled = True
    
    # Clear previous outputs
    out_research.clear_output()
    out_script.clear_output()
    out_seo.clear_output()
    
    # --- Phase 1: Research ---
    with out_research:
        print(f"ğŸ”� Researching '{topic}'...")
        res_data = run_research_agent(topic)
        print(res_data)
        
    # --- Phase 2: Script ---
    with out_script:
        print("âœ�ï¸� Writing Script...")
        script_data = run_script_agent(res_data)
        print(script_data)
        
    # --- Phase 3: SEO ---
    with out_seo:
        print("ğŸ“ˆ Optimizing SEO...")
        seo_data = run_seo_agent(script_data)
        print(seo_data)
    
    run_btn.description = 'Generate Video Package ğŸš€'
    run_btn.disabled = False
    
    # Open the first tab
    display(HTML("<b>âœ… Done! Click the tabs below to see results.</b>"))

run_btn.on_click(on_click)

# --- Display ---
display(header, topic_input, run_btn, tabs)

