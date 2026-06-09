# Agents for Good: Bio-Sentinel Multi-Agent System - Algal Bloom Predictor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import time
import json # Required for simulating LLM interaction

N_SAMPLES = 1000
np.random.seed(42)

data = pd.DataFrame({
    'Temperature_C': np.random.normal(24, 4, N_SAMPLES),
    'pH': np.random.normal(7.8, 0.5, N_SAMPLES),
    'Total_Nitrogen_mgL': np.random.lognormal(mean=0.5, sigma=0.4, size=N_SAMPLES),
    'Total_Phosphorus_ugL': np.random.lognormal(mean=3.5, sigma=0.5, size=N_SAMPLES),
    'Dissolved_Oxygen_mgL': np.random.normal(7.5, 1.5, N_SAMPLES)
})

tn_norm = (data['Total_Nitrogen_mgL'] - data['Total_Nitrogen_mgL'].min()) / (data['Total_Nitrogen_mgL'].max() - data['Total_Nitrogen_mgL'].min())
tp_norm = (data['Total_Phosphorus_ugL'] - data['Total_Phosphorus_ugL'].min()) / (data['Total_Phosphorus_ugL'].max() - data['Total_Phosphorus_ugL'].min())
temp_norm = (data['Temperature_C'] - data['Temperature_C'].min()) / (data['Temperature_C'].max() - data['Temperature_C'].min())
ph_factor = np.clip((data['pH'] - 7.5) / 1.5, 0, 1)

risk_score = (2 * tn_norm + 3 * tp_norm + 1.5 * temp_norm + 1 * ph_factor) / 7.5
risk_score += np.random.normal(0, 0.1, N_SAMPLES)

BLOOM_THRESHOLD = 0.65
data['Algal_Bloom_Risk'] = (risk_score > BLOOM_THRESHOLD).astype(int)

# --- 2. MODEL TRAINING (ML Predictor Agent's Core) ---
X = data.drop('Algal_Bloom_Risk', axis=1)
y = data['Algal_Bloom_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


# --- 3. AGENT DEFINITIONS (Multi-Agent System) ---

def remediation_planner_agent(sensor_data, risk_proba):
    """
    Simulates an LLM call to generate a specific, actionable response plan.
    It uses a system prompt for persona and requests grounding via Google Search 
    (simulated here by including the 'tools' property in the simulated payload).
    """
    temp, ph, tn, tp, do = sensor_data
    
    
    system_prompt = "Act as an emergency environmental response coordinator. Based on the data provided, formulate a concise, three-step action plan to mitigate the harmful algal bloom risk. Prioritize public safety and source your steps with real-world methods."
    
    
    query_message = f"""
    ML Prediction: HIGH RISK ({risk_proba * 100:.2f}% probability). 
    Key Factors: Total Phosphorus (TP) is critically high at {tp:.2f} Âµg/L, Total Nitrogen (TN) is {tn:.2f} mg/L, and water Temperature is warm at {temp:.1f}Â°C.
    Generate a plan to prevent a full bloom. Include a search step for 'most effective methods to reduce phosphorus in freshwater lakes'.
    """
    
    
    print("ğŸ§  Remediation Planner Agent: Querying LLM for actionable plan...")
    
    simulated_payload = {
        "contents": [{"parts": [{"text": query_message}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}], # Signifies the agent uses the Google Search tool
        "model": "gemini-2.5-flash-preview-09-2025"
    }


    simulated_llm_text = f"""
    **Immediate Action Plan (HIGH RISK)**

    1.  **Public Health Alert:** Issue a Level 2 (High Risk) advisory for the affected water body. Specifically warn against recreational contact and pet consumption due to potential cyanotoxins.
    2.  **Nutrient Mitigation (TP Focus):** Initiate a treatment protocol focused on phosphorus inactivation, such as applying aluminum sulfate (alum) to bind and precipitate TP from the water column. Dosage should be calculated based on the high TP reading of {tp:.2f} Âµg/L.
    3.  **Aeration and Circulation:** Deploy or increase operation of existing aeration systems (e.g., surface aerators or hypolimnetic oxygenation) to boost Dissolved Oxygen (currently {do:.1f} mg/L) and prevent stratification, which accelerates bloom formation.
    """
    
    print("\n" + "="*50)
    print(simulated_llm_text)
    print("="*50)


# --- AGENT 2: ML PREDICTOR AGENT (Sequential) ---
def ml_predictor_agent(sensor_data, trained_model):
    """
    Runs the predictive model on the data received from the Monitor Agent.
    """

    new_data = pd.DataFrame([sensor_data], 
                            columns=['Temperature_C', 'pH', 'Total_Nitrogen_mgL', 'Total_Phosphorus_ugL', 'Dissolved_Oxygen_mgL'])
    
    risk_proba = trained_model.predict_proba(new_data)[:, 1][0]
    prediction = trained_model.predict(new_data)[0]
    
    risk_level = "HIGH" if prediction == 1 else "LOW"
    
    print(f"\nğŸ“Š ML Predictor Agent: Running risk analysis...")
    print(f"Predicted Probability of Bloom: {risk_proba * 100:.2f}%")
    
    if risk_level == "HIGH":
        print("ğŸš¨ ML PREDICTION: HIGH RISK")
        # Pass control to the Remediation Planner Agent
        remediation_planner_agent(sensor_data, risk_proba)
    else:
        print("âœ… ML PREDICTION: LOW RISK. Conditions stable.")
        

# --- AGENT 1: DATA MONITOR AGENT (Loop Agent) ---
def data_monitor_agent(trained_model, iterations=3):
    """
    Simulates the monitoring loop and feeds data to the ML Predictor Agent.
    This demonstrates a Loop Agent concept.
    """
    print(f"ğŸ›°ï¸� Data Monitor Agent: Starting continuous surveillance loop (Simulated {iterations} cycles)...")
    
    # Define a set of simulated sensor readings (3 cycles)
    simulated_sensor_readings = [
        # Cycle 1: Low Risk
        (19.0, 7.0, 0.3, 15.0, 8.5), 
        # Cycle 2: Medium/High Risk
        (26.5, 8.0, 1.2, 50.0, 7.0),
        # Cycle 3: Critical High Risk
        (28.5, 8.5, 2.5, 70.0, 6.2)
    ]
    
    for i, reading in enumerate(simulated_sensor_readings[:iterations]):
        temp, ph, tn, tp, do = reading
        print(f"\n--- MONITORING CYCLE {i+1} ---")
        print(f"Received sensor data: T={temp}Â°C, pH={ph}, TN={tn:.2f} mg/L, TP={tp:.2f} Âµg/L, DO={do:.1f} mg/L")
        
        # Pass data sequentially to the ML Predictor Agent
        ml_predictor_agent(reading, trained_model)
        
        # Simulated delay for continuous monitoring loop
        if i < iterations - 1:
            print("\n[Agent Paused for next cycle... (simulated time gap)]")
            time.sleep(0.1) # Simulate real-time delay

# --- 4. EXECUTION ---
if __name__ == '__main__':
    # Start the multi-agent system orchestrator
    data_monitor_agent(model, iterations=3)

