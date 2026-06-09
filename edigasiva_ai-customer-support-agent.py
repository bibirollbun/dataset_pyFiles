!pip install -U google-generativeai

import google.generativeai as genai
import pandas as pd
import logging

genai.configure(api_key="AIzaSyC5HUQGzRK414JhUEXn7elDk2MbrY-XvXU")

model = genai.GenerativeModel("gemini-1.5-flash-latest")

conversation_memory = []
ticket_db = []

def faq_agent(message):
    logging.info("FAQ Agent Running")
    prompt = f"You are a Customer Support AI. Answer clearly.\nUser: {message}\nResponse:"
    response = model.generate_content(prompt)
    return response.text

def ticket_agent(message):
    logging.info("Ticket Agent Running")
    ticket_id = f"TKT-{len(ticket_db)+1}"
    ticket_db.append({"ticket": ticket_id, "issue": message})
    return f"Ticket created successfully: {ticket_id}"

def summary_agent(response):
    logging.info("Summary Agent Running")
    return f"Summary stored: {response}"

def support_agent(message):
    conversation_memory.append({"user": message})
    escalate = faq_agent(message)

    if "change" in message or "refund" in message:
        final = escalate
    else:
        final = ticket_agent(message)

    summary = summary_agent(final)
    conversation_memory.append({"agent": final})
    return final, summary

samples = [
    "What is your refund policy?",
    "I want to change my delivery address",
    "What about warranty?"
]

for s in samples:
    print("User:", s)
    res, summ = support_agent(s)
    print("Agent:", res)
    print(summ)
    print("-"*40)

print("\nTicket Database:")
display(pd.DataFrame(ticket_db))

print("\nConversation Memory:")
print(conversation_memory)

