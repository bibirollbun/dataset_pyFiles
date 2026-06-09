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


# %% [code]
!pip install -q -U google-generativeai chromadb pandas

import os

# ------------------------------------------------------------------
# ğŸ› ï¸� FIX: Disable Tokenizer Parallelism to prevent deadlock warnings
# This must be run BEFORE importing chromadb or other ML libraries
# ------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import google.generativeai as genai
import textwrap
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------------------------------------------
# CONFIGURATION
# ğŸš¨ IMPORTANT: Set your API key in the Kaggle 'Secrets' add-on
# with the label 'GOOGLE_API_KEY', or set it manually below.
# ------------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
except:
    # If running locally, set your key here
    api_key = "YOUR_GEMINI_API_KEY_HERE"

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

print("Configuration Complete.")


# %% [code]
# We will use a simple in-memory ChromaDB for this agent's Long Term Memory (RAG)

class KnowledgeBase:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        
        # ---------------------------------------------------------
        # ğŸ› ï¸� FIX: Delete the collection if it exists.
        # This prevents "Collection already exists" error on re-runs.
        # ---------------------------------------------------------
        try:
            self.chroma_client.delete_collection(name="cpd_faq")
        except:
            pass # It didn't exist, so we are good to go
            
        self.collection = self.chroma_client.create_collection(name="cpd_faq")
        self._populate_db()

    def _populate_db(self):
        # Raw Data from the provided text
        documents = [
            "Application Requirements: Teaching cert (Ofqual AET/PTTLS), Nursing degree accepted. 12 months exp in advanced treatments, 6 months in foundation. Insurance proof, manuals, policies, logo required.",
            "Accreditation Timeline: 7-10 working days usually, sometimes within a week.",
            "Manual Contents: Health & safety, A&P, Treatment history, Contra-indications, Consultation, Contra-actions, Pre/Post advice, Treatment protocol.",
            "Online Courses: Theory can be online. Practical can be online ONLY IF non-invasive. Invasive (Botox, microneedling) foundation courses cannot be fully online.",
            "Online Course Requirements: Same docs as regular. Downloadable manual or PPT slides/script required.",
            "Required Policies: Privacy, Complaints, Appeals, Equality, GDPR, H&S, Safeguarding, Terms & Conditions, Malpractice, etc.",
            "Accredited Courses: Beauty, Nails, Lash/Brow, Massage, Aesthetics (fillers/botox), Tattooing, Hair, Psychic/New Age, Business courses.",
            "Gold Status: Minimum 20 comments with 95% positive feedback.",
            "Awards: Top Rated (4.5 avg, no complaints), Rising Star (Newcomers), Diamond Academy (1000+ positive feedback).",
            "Complaints: Academy can dispute reviews. Multiple substantiated complaints in 12 months may cancel accreditation.",
            "Certificates: Academy issues them. Must have CPD silver/gold seal.",
            "Experience to Teach: General beauty: 12 months. Aesthetics/Skilled: 1 year experience required.",
            "Reviews: Must be genuine. Models/Clients cannot leave reviews, only students. Posted within 3 days.",
            "Renewal: Invoice sent 2 weeks prior. Starter (1-5 courses) Â£199. Premium (6-10) Â£349. Platinum (11-20) Â£549. Diamond (Unlimited) Â£549 renewal.",
            "Pricing Starter: 1-5 Courses, Â£199/yr. Add courses Â£55 each.",
            "Pricing Premium: 6-10 Courses, Â£349/yr. Add courses Â£44 each.",
            "Pricing Platinum: 11-20 Courses, Â£549/yr. Add courses Â£33 each.",
            "Pricing Diamond: Unlimited Courses, Â£849/yr. Renewal Â£549.",
            "Add Trainers: Free if working under business name."
        ]
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            ids=[f"id_{i}" for i in range(len(documents))]
        )
        print("Knowledge Base Populated.")

    def query(self, question):
        results = self.collection.query(
            query_texts=[question],
            n_results=3
        )
        return results['documents'][0]

# Initialize the KB
kb = KnowledgeBase()


# %% [code]
# --- Tool 1: Pricing Calculator ---
def calculate_accreditation_cost(num_courses: int):
    """
    Calculates the annual plan cost based on the number of courses.
    Use this when a user asks about price, cost, or fees for a specific number of courses.
    """
    if num_courses <= 0:
        return "Number of courses must be at least 1."
    
    # Logic based on provided pricing tiers
    if num_courses <= 5:
        plan = "Starter Plan"
        cost = 199
        per_course_add = 55
        notes = "Includes 1-5 courses. Additional courses are Â£55 each."
    elif num_courses <= 10:
        plan = "Premium Plan"
        cost = 349
        per_course_add = 44
        notes = "Includes 6-10 courses. Additional courses are Â£44 each."
    elif num_courses <= 20:
        plan = "Platinum Plan"
        cost = 549
        per_course_add = 33
        notes = "Includes 11-20 courses. Additional courses are Â£33 each."
    else:
        plan = "Diamond Plan"
        cost = 849
        per_course_add = 0
        notes = "Unlimited courses. Renewal fee is only Â£549."

    return {
        "Recommended_Plan": plan,
        "Annual_Cost_GBP": cost,
        "Notes": notes
    }

# --- Tool 2: Knowledge Retrieval Wrapper ---
def search_regulations_and_faq(query: str):
    """
    Searches the Centre of CPD Excellence knowledge base for policies, 
    accreditation rules, awards, and requirements.
    """
    results = kb.query(query)
    return "\n".join(results)

# Register tools for Gemini
tools_list = [calculate_accreditation_cost, search_regulations_and_faq]


# %% [code]
# System Instruction defining the Persona
system_instruction = """
You are 'CPDAssist', a helpful and professional customer service agent for the 'Centre of CPD Excellence'. 
Your goal is to assist training academies with accreditation, pricing, and regulations.

GUIDELINES:
1. ALWAYS use the 'search_regulations_and_faq' tool to verify policy questions (especially about online courses and invasive procedures).
2. ALWAYS use the 'calculate_accreditation_cost' tool if the user mentions a specific number of courses to get the exact price.
3. Be polite, British English spelling (e.g., 'programme', 'centre').
4. If a user asks about "Online Botulinum Toxin" or invasive procedures, strictly warn them based on the FAQ that foundation courses cannot be fully online.
5. Keep answers concise but informative.
"""

# Initialize Model with Tools
# ğŸ› ï¸� FIX APPLIED: Changed 'gemini-1.5-flash' to the stable 'gemini-2.5-flash'
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash', 
    tools=tools_list,
    system_instruction=system_instruction
)

# Start Chat Session (Automatic History Management)
chat = model.start_chat(enable_automatic_function_calling=True)

def agent_query(user_input):
    print(f"\nğŸ‘¤ User: {user_input}")
    response = chat.send_message(user_input)
    # The SDK automatically handles the tool calls behind the scenes
    # and returns the final synthesized text.
    print(f"ğŸ¤– CPDAssist: {response.text}")
    print("-" * 50)


# %% [code]
# Demo 1: Pricing Logic (Math Tool)
agent_query("I have 8 courses I want to accredit. How much will that cost?")

# Demo 2: Regulatory Knowledge (RAG Tool)
agent_query("Can I teach a foundation Botulinum Toxin course entirely online via Zoom?")

# Demo 3: General Info (RAG Tool)
agent_query("What do I need to put in my training manuals?")

# Demo 4: Awards Info
agent_query("How do I get the Diamond Academy award?")

