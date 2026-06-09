# Install dependencies (run once)
!pip install openai python-docx duckduckgo-search

from openai import OpenAI
from datetime import datetime

# 1. Client setup with Gemini OpenAI-compatible API
client = OpenAI(
    api_key="AIzaSyB9zSLf54J7NUa32rlgV6r4qbvOAelyAso",  
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. Summarize legal issue with category
def summarize_case(text: str, category: str) -> str:
    prompt = f"""
You are a legal assistant for Indian law.

Category: {category}

1) Identify who is the user (tenant/consumer/employee etc.).
2) Identify the main issue in simple terms.
3) Mention any dates, amounts, or places given.
4) Write in very simple English, 4-5 bullet points.

User's description:
{text}
"""
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 3. Provide step-by-step legal guidance in Indian context
def legal_guide(summary: str, category: str) -> str:
    prompt = f"""
You are an Indian legal information assistant (not a lawyer).

User's case summary:
{summary}

Category: {category}

Write a clear step-by-step guide:

1. IMPORTANT NOTE
   - State clearly this is general info, not legal advice.

2. DOCUMENTS TO COLLECT
   - List proofs, agreements, messages, payments, ID etc.

3. TALKING TO THE OTHER PARTY
   - How to write polite but firm message/email/notice.

4. LEGAL / OFFICIAL OPTIONS
   - Consumer forum / rent authority / police / cyber cell as relevant.
   - Explain what each forum is for.

5. WHEN TO CONSULT A LAWYER
   - Situations to seek professional legal help.

Use simple English and focus on India only.
"""
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. Generate text document with details and disclaimer
def generate_document(user_input: str, summary: str, steps: str, category: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    document = f"""
=== AI LEGAL ASSISTANT REPORT ===
Generated at: {now}
Category: {category}

--- USER DESCRIPTION ---
{user_input}

--- CASE SUMMARY (AI) ---
{summary}

--- STEP-BY-STEP GUIDE (AI) ---
{steps}

--- IMPORTANT DISCLAIMER ---
This document provides general legal information only.
It is NOT a substitute for qualified legal advice.
For serious or urgent matters, please consult a licensed lawyer or authority.
"""
    filepath = "/kaggle/working/legal_document.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(document)
    return filepath

# 5. Safety check for high-risk keywords
def has_high_risk_keywords(text: str) -> bool:
    keywords = ["suicide", "murder", "kill", "rape", "terrorist", "bomb"]
    t = text.lower()
    return any(k in t for k in keywords)

# 6. Main assistant function integrating all
def ai_law_assistant(user_input: str, category: str):
    if has_high_risk_keywords(user_input):
        summary = "The issue involves serious criminal or life-threatening matters."
        guide = (
            "This is an emergency. Immediately contact police, emergency helpline, or a qualified criminal lawyer."
            " Do not rely on AI tools alone."
        )
        doc = generate_document(user_input, summary, guide, category)
        return summary, guide, doc

    summary = summarize_case(user_input, category)
    guide = legal_guide(summary, category)
    doc = generate_document(user_input, summary, guide, category)
    return summary, guide, doc

# Example usage:

user_input = "Explain FIR filing steps for cybercrime."
category = "cybercrime"  # Categories could be landlord_tenant, consumer, online_fraud etc.

summary, guide, doc_file = ai_law_assistant(user_input, category)

print("=== SUMMARY ===")
print(summary)

print("\n=== LEGAL GUIDE ===")
print(guide)

print("\n=== DOCUMENT GENERATED ===")
print(doc_file)


