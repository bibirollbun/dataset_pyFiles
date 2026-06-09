# Install libraries for PDF, ADK, and Graphing (for the visual aid)
!pip install google-adk pdfplumber networkx matplotlib --quiet
# We don't need google-cloud-aiplatform anymore!

import sys
if "google.colab" in sys.modules:
    from google.colab import userdata
    api_key = userdata.get('GOOGLE_API_KEY')
else:
    # For Kaggle Secrets
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")

# Set the API Key for the environment
import os
os.environ["GOOGLE_API_KEY"] = api_key

print("âœ… API Key loaded. We are ready to go without Google Cloud!")


import pdfplumber
import networkx as nx
import matplotlib.pyplot as plt
import os

# --- TOOL 1: PDF Reader ---
def read_iep_pdf(file_path: str) -> str:
    """Reads a local IEP PDF file and extracts text."""
    print(f"DEBUG: ğŸ“‚ Reading PDF: {file_path}")
    if not os.path.exists(file_path):
        return "Error: File not found."
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    return text[:10000]

# --- TOOL 2: Local Concept Map Generator (Replaces Imagen) ---
def generate_concept_map(nodes_and_edges: str) -> str:
    """
    Generates a visual flowchart/concept map and saves it as an image.
    Args:
        nodes_and_edges: A string list of edges in format 'Start->End, A->B'. 
                         Example: 'Sun->Evaporation, Evaporation->Clouds'
    """
    print(f"DEBUG: ğŸ�¨ Drawing chart for: {nodes_and_edges}")
    try:
        G = nx.DiGraph()
        
        # Parse the simple string format 'A->B, C->D'
        pairs = [pair.strip() for pair in nodes_and_edges.split(',')]
        for pair in pairs:
            if '->' in pair:
                u, v = pair.split('->')
                G.add_edge(u.strip(), v.strip())
        
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=10, font_weight='bold', 
                arrowsize=20, edge_color='gray')
        
        output_file = "visual_aid.png"
        plt.title("Lesson Concept Map")
        plt.savefig(output_file, format="PNG")
        plt.close()
        
        return f"Diagram saved to {output_file}"
    except Exception as e:
        return f"Drawing failed: {e}"


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini

# Use Flash for high speed and higher rate limits (15 RPM)
flash_config = Gemini(model="gemini-2.5-flash")

# --- Agent 1: Analyst ---
analyst = Agent(
    name="analyst",
    model=flash_config,
    tools=[read_iep_pdf],
    instruction="""
    1. Read the IEP PDF.
    2. Summarize the 'Accommodations' and 'Learning Goals'.
    """
)

# --- Agent 2: Teacher ---
teacher = Agent(
    name="teacher",
    model=flash_config,
    instruction="""
    1. Rewrite the lesson topic based on the accommodations.
    2. Instead of an image description, create a FLOWCHART structure.
    3. Output a line starting with "CHART_DATA:" followed by steps separated by commas and arrows.
    Example: CHART_DATA: Sun->Heat, Heat->Water, Water->Vapor
    """
)

# --- Agent 3: Visualizer (The "Artist") ---
visualizer = Agent(
    name="visualizer",
    model=flash_config,
    tools=[generate_concept_map],
    instruction="""
    1. Find the "CHART_DATA:" line from the teacher.
    2. Call `generate_concept_map` with that data.
    3. Confirm when the file is saved.
    """
)


# First, install the library to create PDFs
!pip install reportlab --quiet

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_dummy_iep(filename="dummy_iep.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # --- Title & Header ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "INDIVIDUALIZED EDUCATION PROGRAM (IEP)")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, 735, "CONFIDENTIAL DOCUMENT - FOR EDUCATIONAL USE ONLY")
    c.line(50, 730, 550, 730) # Horizontal line
    
    # --- Student Information ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 700, "1. STUDENT INFORMATION")
    
    c.setFont("Helvetica", 11)
    c.drawString(70, 680, "Student Name: Alex Taylor")
    c.drawString(70, 665, "Grade: 5")
    c.drawString(300, 680, "DOB: 05/12/2014")
    c.drawString(300, 665, "ID: #882910")
    
    # --- Present Levels (Noise for the agent to filter out) ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 630, "2. PRESENT LEVELS OF PERFORMANCE")
    
    c.setFont("Helvetica", 11)
    text = "Alex is a creative student who excels in art and hands-on activities."
    c.drawString(70, 610, text)
    text2 = "However, Alex struggles with large blocks of text and abstract concepts."
    c.drawString(70, 595, text2)
    
    # --- Accommodations (The KEY part for your agent) ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 550, "3. ACCOMMODATIONS & MODIFICATIONS")
    
    c.setFont("Helvetica", 11)
    # We want the agent to extract these specifically:
    accommodations = [
        "- Text-to-speech for reading passages longer than 3 paragraphs.",
        "- Visual aids (diagrams, charts) required for all new concepts.",
        "- Extended time (1.5x) on written assignments.",
        "- Simplified vocabulary for complex instructions.",
        "- Use of a graphic organizer for writing tasks."
    ]
    
    y_position = 530
    for item in accommodations:
        c.drawString(70, y_position, item)
        y_position -= 20
        
    # --- Goals ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position - 20, "4. ANNUAL GOALS")
    c.setFont("Helvetica", 11)
    c.drawString(70, y_position - 40, "Goal 1: Alex will improve reading comprehension by using visual cues.")
    
    c.save()
    print(f"âœ… '{filename}' has been created successfully!")

# Run the function
create_dummy_iep()


import asyncio
import os
import networkx as nx
import matplotlib.pyplot as plt
import reportlab.pdfgen.canvas as pdf_canvas
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from IPython.display import Image, display

# --- 1. SETUP MODELS ---
# Use Flash for speed and better rate limits
flash_config = Gemini(model="gemini-2.5-flash")

# --- 2. DEFINE AGENTS (Fresh instances every run) ---
analyst = Agent(
    name="analyst",
    model=flash_config,
    tools=[read_iep_pdf], # Ensure 'read_iep_pdf' from Cell 2 is loaded
    instruction="""
    1. Read the IEP PDF.
    2. Summarize the 'Accommodations' and 'Learning Goals'.
    """
)

teacher = Agent(
    name="teacher",
    model=flash_config,
    instruction="""
    1. Rewrite the lesson topic based on the accommodations.
    2. Instead of an image description, create a FLOWCHART structure.
    3. Output a line starting with "CHART_DATA:" followed by steps separated by commas and arrows.
    Example: CHART_DATA: Sun->Heat, Heat->Water, Water->Vapor
    """
)

visualizer = Agent(
    name="visualizer",
    model=flash_config,
    tools=[generate_concept_map], # Ensure 'generate_concept_map' from Cell 2 is loaded
    instruction="""
    1. Find the "CHART_DATA:" line from the teacher.
    2. Call `generate_concept_map` with that data.
    3. Confirm when the file is saved.
    """
)

# --- 3. COORDINATOR ---
# Now we bind the fresh agents to the coordinator
iep_bot = SequentialAgent(
    name="iep_bot",
    sub_agents=[analyst, teacher, visualizer]
)

# --- 4. EXECUTION HELPER (Fixed) ---
async def run_demo():
    # Setup Memory
    session_service = InMemorySessionService()
    
    # FIX 1: Use Keyword Arguments strictly
    await session_service.create_session(
        app_name="app",
        user_id="user1",
        session_id="sess1"
    )
    
    # Check for dummy PDF
    if not os.path.exists("dummy_iep.pdf"):
        c = pdf_canvas.Canvas("dummy_iep.pdf")
        c.drawString(100, 700, "Accommodations: Needs visual flowcharts and simplified text.")
        c.save()
        print("ğŸ“„ Created dummy PDF.")

    # Init Runner
    runner = Runner(agent=iep_bot, app_name="app", session_service=session_service)
    
    # Input
    user_msg = types.Content(role="user", parts=[types.Part(text="File: dummy_iep.pdf. Topic: Photosynthesis.")])
    
    print("ğŸ¤– Agents working (Flash Model)...")
    
    final_text = ""
    
    # FIX 2: Use Keyword Arguments strictly here too
    async for event in runner.run_async(
        user_id="user1",
        session_id="sess1",
        new_message=user_msg
    ):
        if event.is_final_response():
            final_text = event.content.parts[0].text
            
    print("\nâœ… Text Result:\n", final_text)
    
    # Show Image
    if os.path.exists("visual_aid.png"):
        print("\nğŸ�¨ Generated Flowchart:")
        display(Image("visual_aid.png"))
    else:
        print("\nâš ï¸� No diagram generated.")

# --- 5. RUN IT ---
await run_demo()

