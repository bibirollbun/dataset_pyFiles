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


# ================================================================
# CAPSTONE PROJECT – CUSTOMER SUPPORT AI ASSISTANT (SIMULATED AGENTS)
# ================================================================

import random
from dataclasses import dataclass

# ------------------------------------------------
# Example Knowledge Base (Help Articles)
# ------------------------------------------------
HELP_ARTICLES = {
    "double_charge": "If you were charged twice, please check your bank statement. Duplicate charges are automatically refunded within 3–5 business days.",
    "password_reset": "To reset your password, go to the login screen and click 'Forgot Password'.",
    "login_issue": "If you're unable to log in, make sure your email is verified and your internet is stable.",
    "bug_report": "Thanks for reporting a bug. Our engineering team reviews all issues within 24 hours."
}

# ------------------------------------------------
# Agent 1: Classifier Agent
# ------------------------------------------------
@dataclass
class ClassifierAgent:
    def classify(self, message):
        msg = message.lower()

        if "charged" in msg or "billing" in msg or "payment" in msg:
            return "double_charge"
        elif "password" in msg or "reset" in msg:
            return "password_reset"
        elif "log in" in msg or "login" in msg:
            return "login_issue"
        elif "bug" in msg or "error" in msg:
            return "bug_report"
        else:
            return "unknown"


# ------------------------------------------------
# Agent 2: Retrieval Agent
# ------------------------------------------------
@dataclass
class RetrievalAgent:
    knowledge_base: dict

    def retrieve(self, topic):
        return self.knowledge_base.get(topic, "Sorry, I couldn't find an article for this issue.")


# ------------------------------------------------
# Agent 3: Response Agent
# ------------------------------------------------
@dataclass
class ResponseAgent:
    def craft_response(self, user_message, article_text):
        return f"""
Thanks for reaching out!

You said: "{user_message}"

Here’s what I found that may help:
{article_text}

If this doesn’t solve the issue, please let me know and I’ll help further.
"""


# ------------------------------------------------
# AI SYSTEM (Coordinator)
# ------------------------------------------------
class CustomerSupportAI:
    def __init__(self):
        self.classifier = ClassifierAgent()
        self.retriever = RetrievalAgent(HELP_ARTICLES)
        self.responder = ResponseAgent()

    def run(self, message):
        topic = self.classifier.classify(message)
        article = self.retriever.retrieve(topic)
        answer = self.responder.craft_response(message, article)
        return answer


# ------------------------------------------------
# Test the System
# ------------------------------------------------

ai = CustomerSupportAI()

sample_messages = [
    "I was charged twice!",
    "I can't log in to my account.",
    "How do I reset my password?",
    "Is this a bug?",
    "Hello, I need help."
]

for msg in sample_messages:
    print("=" * 60)
    print("USER:", msg)
    print("AI ASSISTANT:", ai.run(msg))





