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


# -------------------------------
# STEP 1: Install latest library
# -------------------------------
!pip install --upgrade google-generativeai

# -------------------------------
# STEP 2: Import and configure API
# -------------------------------
import google.generativeai as genai
import os

# Use environment variable for safety
genai.configure(api_key=os.getenv("AIzaSyD19aEiwIlFk191APisZPxCS-H5balNPyw"))

# -------------------------------
# STEP 3: Select Gemini model
# -------------------------------
import google.generativeai as genai
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# -------------------------------
# STEP 4: Build the prompt
# -------------------------------
def build_prompt(ticket_text):
    return f"""
You are a customer support ticket classifier.
Classify the user's ticket into exactly ONE of these categories:

1. Technical Issue
2. Billing & Payments
3. Account & Access
4. Product/Service Inquiry
5. General Support / Other

Return ONLY the category name.

Ticket:
{ticket_text}
"""

# -------------------------------
# STEP 5: Classify a ticket
# -------------------------------
def classify_ticket(ticket_text):
    prompt = build_prompt(ticket_text)
response = model.generate_content([prompt])
    return response.text.strip()  # returns ONLY the category

# -------------------------------
# STEP 6: List of tickets
# -------------------------------
tickets = [
    "I was charged twice for my subscription this month.",
    "Why did my invoice amount go up suddenly?",
    "Can I get a refund for the mistaken charge?",
    "The billing address on my receipt is incorrect.",
    "My credit card expired; how do I update it?",
    "I need a copy of my invoice for last month.",
    "I was promised a discount but it was not applied.",
    "Why am I seeing an extra fee on my statement?",
    "My subscription renewed without notifying me.",
    "I want to change my billing cycle date.",
    "The payment page is not accepting my card.",
    "I accidentally purchased the wrong plan.",
    "My receipt is missing from the dashboard.",
    "I need to switch from monthly to annual billing.",
    "A charge appeared on my account that I don’t recognize.",
    "My PayPal transaction shows paid but your system says unpaid.",
    "I upgraded my plan but was billed for both plans.",
    "Can you remove tax charges from my invoice?"
]

# -------------------------------
# STEP 7: Classify all tickets
# -------------------------------
for i, ticket in enumerate(tickets, 1):
    category = classify_ticket(ticket)
    print(f"{i}. Ticket: {ticket}")
    print(f"   → Category: {category}\n")


