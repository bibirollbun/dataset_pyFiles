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


Project Overview:

# Ecosort AI â€“ Smart Waste Sorting & Recycling Advisor
 
Problem: Households and organizations often struggle with sorting waste properly for recycling, composting, and disposal. Misclassification leads to environmental harm and inefficient recycling.

Solution: Ecosort AI is a multi-agent system that classifies waste items into categories like Organic, Plastic, Paper, Metal, Glass, and E-waste. It also provides recycling/disposal advice and explanations for the classification.

Value: Helps users quickly sort waste correctly, reduces contamination in recycling streams, and educates about waste categories.



# Project Folder Structure

i created the following folders to organize the project:

- agents : for agent scripts(multi-agent orchestrator,etc.)
- tools : ontains classification and advice tools
- tools/recycling : custom tools for recycling advice
- models : placeholder for future AI models
- memory : session and memory-related code
- logs : optional folder for logging agent activities

This structure helps separate responsibilities, making the code modular and maintainable.


# Create folders for EcoSort AI project

import os

folders = [
    "agents",
    "tools",
    "models",
    "memory",
    "logs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project structure created!")


%%bash
mkdir -p tools/recycling
touch tools/__init__.py
touch tools/recycling/__init__.py


# Tools

1. Waste Classifier (tools/waste_classifier.py):
Classifies waste items into categories based on keywords.

2. Recycling Advice (tools/recycling/advice.py):  
Provides instructions for proper disposal or recycling of each category.

3. Explainability (tools/explainability.py):  
Explains why a specific item falls into a category (educational purpose).

4. Session Memory (session_memory.py): 
Stores the last 5 classified items to provide session context to the user.



%%writefile tools/waste_classifier.py
# Waste Classification Tool (simple rule-based version)

def classify_waste(item: str) -> str:
    """
    Classify the waste item into a basic category.
    This is a simple baseline tool; can be upgraded later.
    """

    item = item.lower()

    # Plastics
    if any(word in item for word in ["plastic", "bottle", "wrapper", "container", "packet"]):
        return "plastic"

    # Paper
    if any(word in item for word in ["paper", "newspaper", "tissue", "cardboard", "box"]):
        return "paper"

    # Metal
    if any(word in item for word in ["can", "tin", "foil", "metal"]):
        return "metal"

    # Glass
    if any(word in item for word in ["glass", "jar", "vial"]):
        return "glass"

    # Organic
    if any(word in item for word in ["banana", "food", "leftover", "fruit", "vegetable", "peel"]):
        return "organic"

    # E-waste
    if any(word in item for word in ["phone", "charger", "battery", "earphones", "electronics"]):
        return "e-waste"

    return "unknown"



from tools.waste_classifier import classify_waste

print(classify_waste("plastic bottle"))
print(classify_waste("banana peel"))
print(classify_waste("aluminum can"))
print(classify_waste("broken glass"))



%%writefile tools/recycling_advice.py


def get_recycling_advice(category: str) -> str:
    """
    Return recycling or disposal advice based on the waste category.
    """
    category = category.lower()

    if category == "plastic":
        return("Plastic items should be rinsed and placed in the plastic recycling bin."
                "Avoid mixing them with organic waste")

    if category == "paper":
        return("Paper and cardboard can be recycled. Make sure they are not wet or oily before disposal.")

    if category == "metal":
        return("Metal cans and foil should be cleaned and placed in the metal recycling bin.")

    if category == "glass":
        return("Glass bottles and jars can be recycled. Avoid breaking them while disposing.")

    if category == "organic":
        return("Organic waste like fruit peels or leftovers should be placed in the compost or wet waste bin.")

    if category == "e-waste":
        return("E-waste like batteries, chargers, and electronics must be taken to a certified e-waste center.")

    return("Category unknown. Please check the item or dispose of it separately.")


%%writefile ecosort_agent.py

from tools.waste_classifier import classify_waste
from tools.recycling_advice import get_recycling_advice
from IPython.display import Markdown, display 

def ecosort_agent(user_input: str) -> str:
    """
    Main agent that uses tools to:
    1) classify the waste item
    2) give recycling advice
    3) return a clean final message
    """

    item = user_input.lower().strip()
    category = classify_waste(item)
    advice = get_recycling_advice(category)

    # prepare final response
    response = (
        f"\u267Bï¸� **Ecosort AI Result**\n\n"
        f"**Item Detected:** {item}\n"
        f"**Category:** {category}\n\n"
        f"**Recycling / Disposal Advice:**\n{advice}"
    )

    return response



from ecosort_agent import ecosort_agent
from IPython.display import Markdown, display

result = ecosort_agent("banana peel")
display(Markdown(result))


from ecosort_agent import ecosort_agent
from IPython.display import Markdown, display

def ecosort_ui():
    print("\u267B Welcome to Ecosort AI! Type 'exit' to quit.")
    
    while True:
        # Take input from the user
        item = input("Enter a waste item: ").strip()
        
        if item.lower() == "exit":
            print("Goodbye!")
            break
        
        # Get the result from your agent
        result = ecosort_agent(item)
        
        # Display the result nicely
        display(Markdown(result))

# Run the UI
ecosort_ui()



%%bash
mkdir -p tools
touch tools/__init__.py



%%writefile tools/waste_classifier.py
def classify_waste(item: str) -> str:
    item = item.lower()
    if any(word in item for word in ["banana", "apple", "food", "peel"]):
        return "organic"
    if any(word in item for word in ["plastic", "bottle", "wrapper"]):
        return "plastic"
    if any(word in item for word in ["metal", "can", "tin"]):
        return "metal"
    if any(word in item for word in ["glass", "jar"]):
        return "glass"
    if any(word in item for word in ["paper", "newspaper", "card"]):
        return "paper"
    return "other"



%%writefile tools/recycling/advice.py
def get_recycling_advice(category: str) -> str:
    tips = {
        "organic": "Put organic waste in the compost bin.",
        "plastic": "Clean and dry plastic before recycling.",
        "metal": "Rinse metal cans and place in recycling.",
        "glass": "Recycle only clean and unbroken glass.",
        "paper": "Dry, clean paper should be recycled.",
        "other": "Dispose in general waste."
    }
    return tips.get(category.lower(), "No advice available.")



%%writefile tools/explainability.py
def explain_classification(item: str, category: str) -> str:
    """
    Returns a brief explanation for why the item is classified in this category.
    """
    explanations = {
        "organic": "Items like food waste, peels, and biodegradable materials are classified as organic because they decompose naturally.",
        "plastic": "Plastic items are made of synthetic polymers, which are non-biodegradable and must be recycled separately.",
        "metal": "Metal objects like cans, foil, and tins are recyclable if clean and sorted properly.",
        "glass": "Glass containers are recyclable but should be unbroken and clean before recycling.",
        "paper": "Paper products come from processed wood pulp and are easily recyclable unless contaminated with oil or moisture.",
        "e-waste": "Electronic waste includes gadgets, batteries, and cables, which should be taken to certified e-waste centers.",
        "other": "Items that do not fit known categories are classified as 'other'."
    }
    return explanations.get(category.lower(), "No explanation available for this category.")




%%writefile session_memory.py

class SessionMemory:
    def __init__(self, limit=5):
        self.limit = limit
        self.history = []

    def add_entry(self, item, category):
        self.history.append({"item": item, "category": category})

        if len(self.history) > self.limit:
            self.history.pop(0)

    def get_history(self):
        return self.history




# Multi-Agent Pipeline (`multi_agent_pipeline.py`)

The pipeline orchestrates agents sequentially:

1. Classification Agent â€“ Identifies the category of the input item.  
2. Advice Agent â€“ Provides disposal/recycling instructions.  
3. Explainability Agent â€“ Gives a short explanation of the classification.  
4. Memory Agent â€“ Saves the item and category in session memory.  
5. Final Response Agent â€“ Combines all outputs into a formatted message.

This modular approach allows easy updates and potential parallelization in future versions.



%%writefile multi_agent_pipeline.py

from tools.waste_classifier import classify_waste
from tools.recycling.advice import get_recycling_advice
from tools.explainability import explain_classification
from session_memory import SessionMemory

# single SessionMemory instance for the notebook runtime
memory = SessionMemory()

# Agent 1 - Classification Agent
def agent_classifier(item: str) -> dict:
    category = classify_waste(item)
    return {"category": category}

# Agent 2 - Recycling Advice Agent
def agent_recycling_advice(category: str) -> dict:
    advice = get_recycling_advice(category)
    return {"advice": advice}

# Agent 3 - Explainability Agent
def agent_explainability(item: str, category: str) -> dict:
    explanation = explain_classification(item, category)
    return {"explanation": explanation}

# Agent 4 - Final Response Agent 
def agent_final_response(item: str, category: str, advice: str, explanation: str, history) -> str:
   
    if history:
        history_lines = [f"- {h['item']}  â†’  {h['category']}" for h in history]
        history_str = "\n".join(history_lines)
    else:
        history_str = "No previous items."

    return (
        f"â™» **Ecosort AI Result**\n\n"
        f"**Item Detected:** {item}\n\n"
        f"**Category:** {category}\n\n"
        f"### ğŸ”� Why this category?\n"
        f"{explanation}\n\n"
        f"### ğŸ—‘ Recycling / Disposal Advice:\n"
        f"{advice}\n\n"
        f"---\n"
        f"### ğŸ“� Session Memory \n"
        f"{history_str}"
    )

# Main orchestrator: runs agents sequentially
def ecosort_multi_agent(item: str) -> str:
    item_clean = item.lower().strip()

    # 1) classify
    c = agent_classifier(item_clean)
    category = c["category"]

    # 2) advice
    a = agent_recycling_advice(category)
    advice = a["advice"]

    # 3) explainability
    e = agent_explainability(item_clean, category)
    explanation = e["explanation"]

    # 4) save to memory
    memory.add_entry(item_clean, category)

    # 5) format final output
    return agent_final_response(item_clean, category, advice, explanation, memory.get_history())




from multi_agent_pipeline import ecosort_multi_agent
from IPython.display import Markdown, display




display(Markdown(ecosort_multi_agent("banana peel")))
display(Markdown(ecosort_multi_agent("plastic bottle")))
display(Markdown(ecosort_multi_agent("newspaper")))


from multi_agent_pipeline import ecosort_multi_agent
from IPython.display import Markdown, display

def ecosort_ui():
    print("â™» Welcome to Ecosort AI â€“ Waste Classification")
    print("Type 'exit' to quit.\n")

    while True:
        item = input("Enter a waste item: ").strip()
        
        if item.lower() == "exit":
            print("Goodbye! ğŸ‘‹")
            break
        
        # Get output from your multi-agent pipeline
        result = ecosort_multi_agent(item)
        
        # Display nicely formatted in notebook
        display(Markdown(result))
        print("\n" + "-"*50 + "\n")

# Run the UI
ecosort_ui()



#Architecture diagram

[User Input] 
       â†“
[Agent 1: Waste Classifier]
       â†“
[Agent 2: Recycling Advice]
       â†“
[Agent 3: Explainability]
       â†“
[Agent 4: Final Response]
       â†“
[Session Memory]


