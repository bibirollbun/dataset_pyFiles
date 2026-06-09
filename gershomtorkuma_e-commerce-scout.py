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


# 1. Install (course-recommended – no LangChain)
!pip install -q --no-cache-dir google-adk==1.0.0 google-generativeai==0.8.3

print("ADK installed – ready!")


# 2. Course-exact imports (from Day 4 multi-agent tutorial)
import json
import uuid
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # For Content and Part

# Debugging: Setup logging for observability (Day 2 lesson)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("ADK multi-agent components imported successfully.")


# 3. Load your API keys (Kaggle Secrets – do NOT hardcode!)
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"]   = user_secrets.get_secret("TAVILY_API_KEY")

print("API keys loaded")


# 4. Three LlmAgents (with output_key for shared state – course Day 4, tool-free)
# FIXED: Use "gemini-2.5-flash" (stable for v1beta API, Day 1: Models lesson)
gemini_model = Gemini(model="gemini-2.5-flash", temperature=0.0)

# Agent 1: Researcher (saves stores to state['stores'])
researcher = LlmAgent(
    model=gemini_model,
    name="researcher",
    instruction="You are a researcher. Given a product, list exactly 4 major online stores that sell it. Format: Store Name - Full URL (one per line).",
    output_key="stores"  # Saves output to state['stores'] for next agent
)

# Agent 2: Checker (uses Gemini knowledge to simulate stock check, saves to state['stock_data'])
checker = LlmAgent(
    model=gemini_model,
    name="checker",
    instruction="You check stock using your knowledge. For each store in state['stores'], estimate availability for the product (In Stock / Out of Stock / Unknown based on general trends). Product: {product}",
    output_key="stock_data"  # Saves to state['stock_data']
)

# Agent 3: Analyst (accesses state['stock_data'], saves to state['recommendation'])
analyst = LlmAgent(
    model=gemini_model,
    name="analyst",
    instruction="You are an inventory manager. Based on state['stock_data'], give a concise recommendation: current status + action (Restock Urgently / Restock Soon / Hold / Reduce Inventory). Product: {product}",
    output_key="recommendation"  # Final output
)

print("3 LlmAgents created with output_keys for shared state")


# 5. SequentialAgent for Multi-Agent Orchestration (course Day 4: SequentialAgent)
# Combines agents into sequential workflow with shared state
inventory_pipeline = SequentialAgent(
    name="inventory_scout_pipeline",
    sub_agents=[researcher, checker, analyst],
    description="Sequential multi-agent for ecommerce stock scouting"
)

# Session service for memory (shared across pipeline)
session_service = InMemorySessionService()

print("SequentialAgent pipeline ready with shared memory!")


# 6. Run Function (course: Create session, then run_async with user_id/session_id/new_message – Day 4 Runner lesson)
async def run_inventory_check(product: str):
    print(f"\nChecking stock for: {product}")
    print("="*70)
    
    user_id = "user"  # Course example: fixed user ID
    session_id = str(uuid.uuid4())
    app_name = "inventory_scout"
    
    # FIXED: Create session with await (fixes "Session not found" – Day 3 Sessions lesson)
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    logger.info(f"Session created: {session_id}")
    
    runner = Runner(
        agent=inventory_pipeline,
        session_service=session_service,
        app_name=app_name
    )
    
    try:
        # FIXED: Use run_async (async generator) with user_id, session_id, new_message (Day 4: Runner lesson)
        content = types.Content(role='user', parts=[types.Part(text=f"Scout inventory for product: {product}")])
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            logger.info(f"Event: {event}")  # Observability: Log events (Day 2 lesson)
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_response = event.content.parts[0].text if event.content and event.content.parts else "No response"
                break
        else:
            final_response = "No final response generated"
        
        # FIXED: Access shared state from updated session (Day 4: Shared State lesson)
        updated_session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        state = updated_session.state if updated_session else {}
        stores = state.get("stores", "N/A")
        stock_data = state.get("stock_data", "N/A")
        recommendation = state.get("recommendation", final_response.strip())
        
        print("Stores found:")
        print(stores)
        print("\nStock data:")
        print(stock_data)
        print("\nFINAL RECOMMENDATION")
        print("="*70)
        print(recommendation)
        
        return {
            "product": product,
            "stores": stores,
            "stock_data": stock_data,
            "recommendation": recommendation
        }
    except Exception as e:
        logger.error(f"Pipeline error: {e}")  # Monitoring: Log errors (Day 2 lesson)
        print(f"Pipeline error: {e}")
        return {"product": product, "recommendation": f"Error: {e}"}

# Run one product (top-level await)
await run_inventory_check("iPhone 16 Pro Max")


# 7. Batch Run (multiple products)
products = [
    "MacBook Air M3",
    "Sony PlayStation 5 Slim",
    "AirPods Pro 2",
    "Nintendo Switch OLED"
]

results = []
for p in products:
    res = await run_inventory_check(p)
    results.append(res)


# 8. Workflow Diagram with matplotlib (capstone: graph visualization)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Boxes for sequential flow
boxes = [
    (1, 8, "User Input\n(Product)", "lightblue"),
    (3, 8, "Researcher Agent\n(Gemini + output_key='stores')", "lightgreen"),
    (6, 8, "Checker Agent\n(Gemini Knowledge + output_key='stock_data')", "lightyellow"),
    (9, 8, "Analyst Agent\n(Gemini + output_key='recommendation')", "lightcoral"),
    (6, 3, "Final Report\n(Shared State Access)", "gold")
]

for x, y, text, color in boxes:
    rect = patches.FancyBboxPatch((x-1, y-1), 3, 1.8, boxstyle="round,pad=0.5", fc=color, ec="black", lw=2.5)
    ax.add_patch(rect)
    ax.text(x+0.5, y, text, ha='center', va='center', fontsize=11, fontweight='bold')

# Arrows for sequential flow
arrows = [(3.5,8,5.5,8), (6.5,8,8.5,8), (6.5,7,6.5,4.5)]
for x1,y1,x2,y2 in arrows:
    arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle='->', mutation_scale=25, lw=4, color='darkblue')
    ax.add_patch(arrow)

ax.text(6, 10, "Sequential Multi-Agent Workflow (SequentialAgent + Shared State)", ha='center', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()


# 9. Summary Table with pandas (data presentation)
df = pd.DataFrame(results)[['product', 'recommendation']]
print("\nINVENTORY RECOMMENDATIONS SUMMARY")
display(df.style.set_caption("Ecommerce Inventory Scout Results")\
    .set_properties(**{'text-align': 'left', 'font-size': '12pt'})\
    .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16pt'), ('font-weight', 'bold')]}]))


# Bonus: Run multiple products
products_to_check = [
    "MacBook Air M3",
    "Sony PlayStation 5 Slim",
    "Nintendo Switch OLED",
    "AirPods Pro 2"
]

for item in products_to_check:
    run_inventory_check(item)

