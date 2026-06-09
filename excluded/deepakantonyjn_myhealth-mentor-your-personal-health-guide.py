!pip install -q -U google-generativeai googlesearch-python


try:
    import google.generativeai as genai
    from googlesearch import search
    import pandas as pd
    print("âœ… SUCCESS: All libraries are working correctly!")
except ImportError as e:
    print(f"â�Œ ERROR: {e}")


import os
import pandas as pd
import time
from google.generativeai import configure, GenerativeModel
from googlesearch import search
from kaggle_secrets import UserSecretsClient

# --- 1. CONFIGURATION ---
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    configure(api_key=api_key)
    print("âœ… API Key successfully loaded from Secrets!")
except Exception as e:
    print(f"â�Œ Error loading key: {e}")
    print("Did you name your secret 'GOOGLE_API_KEY' in the Add-ons menu?")

# --- 2. GENERATE SYNTHETIC DATA ---
# We create a specific scenario: The user is working out hard (High Steps)
# but sleeping poorly and stressed, leading to weight stagnation.
data = {
    'date': ['2025-11-20', '2025-11-21', '2025-11-22', '2025-11-23', '2025-11-24'],
    'steps': [10200, 11500, 9800, 12000, 11000],        # Very Active
    'sleep_hours': [5.0, 4.5, 5.0, 4.0, 5.0],           # Very Poor Sleep
    'stress_level': ['High', 'High', 'Medium', 'High', 'High'],
    'weight_kg': [75.0, 75.1, 75.1, 75.2, 75.2]         # No weight loss (Anomaly)
}

# Save to a CSV file in the notebook environment
df = pd.DataFrame(data)
df.to_csv('health_log.csv', index=False)
print("âœ… Synthetic Health Data Created: 'health_log.csv'")
print(df)


# --- AGENT 1: THE DATA SCIENTIST (Logic & Math) ---
def agent_analyst(file_path):
    print("\nğŸ“Š AGENT 1 (Analyst): Scanning data for anomalies...")
    df = pd.read_csv(file_path)
    
    # Logic: precise calculation using Python
    avg_steps = df['steps'].mean()
    avg_sleep = df['sleep_hours'].mean()
    weight_change = df.iloc[-1]['weight_kg'] - df.iloc[0]['weight_kg']
    
    # Anomaly Detection Logic
    anomaly_report = ""
    if avg_steps > 9000 and weight_change >= 0:
        anomaly_report = (
            f"DETECTED PARADOX: User is highly active (Avg {int(avg_steps)} steps) "
            f"but weight is stagnant/increasing (+{weight_change}kg). "
        )
    
    if avg_sleep < 6:
        anomaly_report += f"POTENTIAL ROOT CAUSE: Severe sleep deprivation (Avg {avg_sleep} hrs)."
        
    print(f"   -> Finding: {anomaly_report}")
    return anomaly_report

# --- AGENT 2: THE RESEARCHER (Tool: Google Search) ---
def agent_researcher(anomaly_context):
    print("\nğŸ”� AGENT 2 (Researcher): verifying medical correlation on Google...")
    
    if not anomaly_context:
        return "No anomalies to research."

    # Create a search query based on the Analyst's finding
    query = f"why does lack of sleep cause weight gain despite exercise scientific mechanism"
    print(f"   -> Searching for: '{query}'")
    
    search_results = []
    try:
        # Perform real Google Search
        for result in search(query, num_results=3, advanced=True):
            search_results.append(f"Source: {result.title}\nSummary: {result.description}")
    except Exception as e:
        search_results.append("Search tool failed. Using general medical knowledge.")

    knowledge_block = "\n\n".join(search_results)
    print("   -> Research Complete.")
    return knowledge_block

# --- AGENT 3: THE CHIEF RESIDENT (Synthesis & Advice) ---
# --- AGENT 3: THE CHIEF RESIDENT (Synthesis & Advice) ---
def agent_advisor(user_data_insight, scientific_proof):
    print("\nğŸ‘¨â€�âš•ï¸� AGENT 3 (Advisor): Synthesizing personal plan...")
    
    # UPDATED: Using the powerful 2.5 Flash model from your list
    model = GenerativeModel('gemini-2.5-flash') 
    
    prompt = f"""
    You are a Senior Metabolic Health Doctor.
    
    THE PATIENT DATA:
    {user_data_insight}
    
    THE MEDICAL SCIENCE:
    {scientific_proof}
    
    TASK:
    Write a direct, empathetic message to the patient.
    1. Validate their hard work (the steps).
    2. Explain the biology: HOW is the sleep deprivation blocking the fat loss?
    3. Give ONE specific prescription (e.g., "Stop the morning run, sleep an extra hour").
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback to 2.0 if 2.5 has a momentary hiccup
        print(f"   -> Switching to fallback model due to: {e}")
        fallback = GenerativeModel('gemini-2.0-flash')
        return fallback.generate_content(prompt).text


def run_bio_detective():
    print("--- ğŸ�¥ STARTING BIO-METRIC DETECTIVE SYSTEM ---")
    
    # Step 1: Analyze the CSV
    problem_statement = agent_analyst('health_log.csv')
    
    # Step 2: Research the specific medical issue
    if problem_statement:
        medical_context = agent_researcher(problem_statement)
        
        # Step 3: Generate Advice
        final_diagnosis = agent_advisor(problem_statement, medical_context)
        
        print("\n" + "="*50)
        print("ğŸ“� FINAL DIAGNOSIS & PLAN")
        print("="*50)
        print(final_diagnosis)
    else:
        print("User is healthy. No intervention needed.")

# Run it
run_bio_detective()


import google.generativeai as genai
print("Available Models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")

