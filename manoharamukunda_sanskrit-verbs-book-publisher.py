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


# Setup your Google Cloud Project / Gemini API Key
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")

# Global Constants
MODEL_NAME = "gemini-2.5-flash-lite"


# --- AGENT DEFINITIONS ---

lead_dhatu_agent = LlmAgent(
    name="lead_dhatu_agent",
    model=MODEL_NAME,
    description="The primary sanskrit verb or dhatu forms assistant. It collaborates with the user to create a sanskirt verb or dhatu forms.",
    instruction=f"""
    You are a Sanskrit linguistic assistant. Your primary function is to help users create technical blog posts.

    Your workflow is as follows:
    1.  **Outline:** You will generate a Sanskrit dhatu or verb's root, meaning, and conjugation type outline and present it to the user. To do this, use the `robust_dhatu_outliner` tool.
    2.  **Refine:** The user can provide feedback to refine the outline. You will continue to refine the outline until it is approved by the user.
    3.  **Write:** Once the user approves the outline, you will write all the tenses forms of the dhatu or verb. To do this, use the `robust_dhatu_writer` tool. Be then open for feedback.
    4.  **Edit:** After the first draft is written, you will present it to the user and ask for feedback. You will then revise the verb's tenses forms based on the feedback. This process will be repeated until the user is satisfied with the result.
    5.  **Format:** After the final editing feedback, you will format the content in tabular form. To do this, use the `dhatu_form_beautifier` tool.
    5.  **Export:** When the user approves the final version, you will ask for a filename and save the Sanskrit verb forms as an Adobe PDF file. If the user agrees, use the `save_dhatu_forms_to_file` tool to save the blog post.

    If you are asked what your name is, respond with Sanskrit Verbs Book Publisher Agent.
    """,
    sub_agents=[
        robust_dhatu_outliner,
        robust_dhatu_writer,
        dhatu_form_editor,
        dhatu_form_beautifier,
    ],
    tools=[
        FunctionTool(save_dhatu_forms_to_file),
    ],
    output_key="dhatu_outline",
)

root_agent = lead_dhatu_agent


from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import google_search

# from ..config import config
# from ..agent_utils import suppress_output_callback
# from ..validation_checkers import OutlineValidationChecker

robust_dhatu_outliner = SequentialAgent(
    name="robust_dhatu_outliner",
    model=MODEL_NAME,
    tools=[google_search],
    description="A robust Sanskrit dhatu or verb root, meaning, and conjugation type outliner that retries if it fails.",
    instruction=f"""
    You are an expert Sanskrit Linguist.
    1. Receive a verb input (English/Sanskrit).
    2. Use 'google_search' to find the Root (Dhatu), Meaning, and Conjugation Type (Pada) from ashtadhyayi.com.
    3. Determine the required tenses based on the 'Pada' (Parasmaipadi, Atmanepadi, or Ubhayapadi).
    4. Output a structured JSON outline containing: Root, Meaning, Pada, and List of Tenses to generate.
    """,
    output_key="dhatu_outline",
    after_agent_callback=suppress_output_callback,
)



def save_dhatu_forms_to_file(content: str, filename: str = "output.pdf") -> str:
    """
    Saves the formatted Sanskrit verb table to a file (Simulated PDF export).
    """
    print(f"ğŸ’¾ [Tool] Saving content to {filename}...")
    # In real impl, use ReportLab or FPDF to generate actual PDF
    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return f"Success: File saved as {filename}"


save_dhatu_forms_to_file = Tool(
    name="save_dhatu_forms_to_file",
    func=save_dhatu_forms_to_file,
    description="Saves the final tabular verb forms to a PDF file."
)


# ---------------------------------------------------------
# Agent 2: Content Writer (The Core Linguist)
# ---------------------------------------------------------
robust_dhatu_writer = SequentialAgent(
    name="robust_dhatu_writer",
    model=MODEL_NAME,
    tools=[google_search],
    instruction="""
    You are a Sanskrit Grammar Automation Engine.
    1. Receive the outline.
    2. Generate the 3x3 conjugation table for all required Tenses (Lakaras) based on the Pada (Active/Middle/Both).
    3. Order is mandatory: Lat, Lit, Lut, Lrt, Lot, Lan, Lin, Asirlin, Vidhilin, Lun, Lrn.
    4. Output the raw Sanskrit forms clearly labeled by Tense and Voice.
    """
)


# ---------------------------------------------------------
# Agent 3: Editor (Quality Control)
# ---------------------------------------------------------
dhatu_form_editor = LlmAgent(
    name="dhatu_form_editor",
    model=MODEL_NAME,
    instruction="""
    You are a professional Sanskrit Editor.
    1. Receive the raw verb forms.
    2. Revise and correct any grammatical inconsistencies based on the rules of Sanskrit Sandhi and Conjugation.
    3. Output the verified forms.
    """
)


# ---------------------------------------------------------
# Agent 4: Beautifier (Formatter & Publisher)
# ---------------------------------------------------------
dhatu_form_beautifier = LlmAgent(
    name="dhatu_form_beautifier",
    model=MODEL_NAME,
    tools=[save_dhatu_forms_to_file],
    instruction="""
    You are a Publishing Layout Expert.
    1. Take the verified verb data.
    2. Format it into the final, clean, preferred tabular grid (Person vs. Number).
    3. Use the 'save_dhatu_forms_to_file' tool to export the content.
    4. Return the file save confirmation.
    """
)


# @title 6. Execution Block
# Running the agent system with a sample Sanskrit verb.

# --- Test Case 1: Ubhayapadi Verb (Both Voices required) ---
user_input_1 = "Generate the book content for the Dhatu à¤•à¥ƒ (ká¹› - to do)."

print("==================================================")
print(f"ğŸ‘¤ User Request 1: {user_input_1}")
print("==================================================")

# Execute the Lead Agent, which triggers the SequentialAgent workflow
response_1 = lead_dhatu_agent.run(user_input_1)

print("\n--- Final System Response (kr) ---")
print(response_1)
print("-----------------------------------")


# --- Test Case 2: Parasmaipadi Verb (Active Voice only) ---
user_input_2 = "Generate the book content for the Dhatu à¤—à¤®à¥� (gam - to go)."

print("\n==================================================")
print(f"ğŸ‘¤ User Request 2: {user_input_2}")
print("==================================================")

response_2 = lead_dhatu_agent.run(user_input_2)

print("\n--- Final System Response (gam) ---")
print(response_2)
print("-----------------------------------")

