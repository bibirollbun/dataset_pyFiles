# --- INSTALL LIBRARIES ---
!pip install -q streamlit langchain langchain-openai pandas python-dotenv langsmith pydantic

# --- WRITE THE APP FILE ---
import os

# Create the app.py file inside the Kaggle environment
code = """
import streamlit as st
import pandas as pd
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# NOTE: In a real Kaggle run, keys are set in the Secrets add-on.
# For this code dump, we assume environment variables are loaded.

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MEMORY_FILE = 'agent_memory.json'

class SearchCriteria(BaseModel):
    budget: int = Field(description="The maximum budget in INR (default 300)")
    provider: str = Field(description="The preferred telecom provider name or 'Any'")

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"history": []}
    return {"history": []}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f, indent=4)

def fetch_telecom_plans():
    data = [
        {"provider": "Jio", "price": 239, "validity": 28, "data_per_day": 1.5, "extras": "Unlimited 5G"},
        {"provider": "Airtel", "price": 265, "validity": 28, "data_per_day": 1.0, "extras": "Hello Tunes"},
        {"provider": "Airtel", "price": 299, "validity": 28, "data_per_day": 1.5, "extras": "Free OTT"},
        {"provider": "Jio", "price": 299, "validity": 28, "data_per_day": 2.0, "extras": "Unlimited 5G"},
        {"provider": "Vi", "price": 299, "validity": 28, "data_per_day": 1.5, "extras": "Data Rollover"},
        {"provider": "Jio", "price": 666, "validity": 84, "data_per_day": 1.5, "extras": "Unlimited 5G"},
    ]
    return pd.DataFrame(data)

def parse_user_requirement(user_input, user_history):
    prompt = ChatPromptTemplate.from_template(
        "Extract MAX BUDGET (integer) and PROVIDER (string). History: {history}. Request: {text}. If missing, assume 300 and 'Any'."
    )
    chain = prompt | llm.with_structured_output(SearchCriteria)
    history_summary = str(user_history['history'][-2:]) 
    result = chain.invoke({"text": user_input, "history": history_summary})
    return result.model_dump()

st.title("ðŸ“¡ Personalized Telecom Recharge Agent")
user_memory = load_memory()
user_input = st.text_area("Describe your needs:", "Budget 300, need Jio")

if st.button("Find Plan"):
    parsed = parse_user_requirement(user_input, user_memory)
    st.success(f"Parsed: Budget â‚¹{parsed['budget']}, Provider: {parsed['provider']}")
    plans = fetch_telecom_plans()
    filtered = plans[plans['price'] <= parsed['budget']]
    if parsed['provider'] != "Any":
        filtered = filtered[filtered['provider'].str.contains(parsed['provider'], case=False)]
    
    if not filtered.empty:
        st.table(filtered)
        user_memory['history'].append({"input": user_input, "rec": int(filtered.iloc[0]['price'])})
        save_memory(user_memory)
    else:
        st.error("No plans found.")
"""

with open("app.py", "w") as f:
    f.write(code)

print("Agent Code Successfully Written to app.py")

