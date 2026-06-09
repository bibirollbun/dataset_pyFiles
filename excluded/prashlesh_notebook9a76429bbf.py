# 3rd-party libraries for the agent
!pip install -q google-genai python-dotenv requests


import os, pathlib, textwrap, sys

# Public GitHub repo for this project
REPO_URL = "https://github.com/vengeanceprashlesh/AI-Researcher.git"

project_root = pathlib.Path.cwd() / "AI-Researcher"
if not project_root.exists():
    !git clone {REPO_URL} {project_root.name}

# Change working directory to the project so imports work
os.chdir(project_root)
sys.path.append(str(project_root))

print("Project root:", project_root)
print("Files in project root:")
for p in project_root.iterdir():
    print(" -", p.name)


import os

# We never hard-code the key in the notebook.
# For interactive runs, you can TEMPORARILY set it, then remove it before saving.
if "GOOGLE_API_KEY" in os.environ:
    print("GOOGLE_API_KEY is set in the environment (good).")
else:
    print(
        "GOOGLE_API_KEY is NOT set.\n"
        "For interactive testing only, you may run:\n"
        "  os.environ['GOOGLE_API_KEY'] = 'YOUR_KEY_HERE'\n"
        "but REMOVE that line before saving the notebook.\n"
    )


from agents import OrchestratorAgent
from memory_manager import ResearchMemoryManager


from pprint import pprint

if not os.getenv("GOOGLE_API_KEY"):
    print("Skipping live demo because GOOGLE_API_KEY is not set.")
else:
    # Initialize memory and orchestrator
    memory = ResearchMemoryManager()
    session_id = memory.start_research_session("kaggle_notebook_demo")
    orchestrator = OrchestratorAgent()

    topic = "Impact of AI on education"

    print(f"ğŸ”¬ Running deep research on: {topic}\n")
    results = orchestrator.deep_research(topic)

    # Save to memory
    memory.save_research_to_session(topic, results)

    # Show summary
    print("\nğŸ“� Summary (first 800 characters):\n")
    summary_text = results["summary"]["summary"]
    print(summary_text[:800] + ("..." if len(summary_text) > 800 else ""))

    # Show basic report info
    if "report" in results:
        print("\nâœ�ï¸� Report word count:", results["report"]["word_count"])

    # Show memory statistics
    ctx = memory.get_research_context()
    stats = ctx["statistics"]
    print("\nğŸ“Š Memory statistics:")
    pprint(stats)

    memory.end_research_session()

