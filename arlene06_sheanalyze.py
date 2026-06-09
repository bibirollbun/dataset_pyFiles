!pip install google-generativeai



import google.generativeai as genai
import json



genai.configure(api_key="Your_API_HEY")
model = genai.GenerativeModel("gemini-2.5-flash")



import json

def safe_json_extract(text):
    """
    Safely extracts JSON content from a model response.
    """
    try:
        return json.loads(text)
    except:
        # Remove possible markdown code fences
        cleaned = (
            text.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        try:
            return json.loads(cleaned)
        except:
            return {"error": "Could not decode JSON", "raw_output": text}



def medical_risk_agent(input_data, condition_name):
    """
    LLM Agent that evaluates the medical risk for one condition.
    """

    prompt = f"""
    You are a medical evaluation agent.
    Assess the risk for: {condition_name}

    User health details:
    {json.dumps(input_data, indent=2)}

    Return ONLY a JSON dictionary like this:

    {{
      "label": "Low / Moderate / High",
      "probability": 0.00,
      "reasons": ["..."],
      "next_steps": ["..."]
    }}
    """
    
    response = model.generate_content(prompt)
    return safe_json_extract(response.text)

    


def run_pipeline(input_data):
    conditions = [
        "PCOS",
        "Thyroid disorder",
        "Infertility",
        "Menopause"
    ]
    
    results = {}
    
    for condition in conditions:
        print(f"Running agent for {condition}...")
        results[condition] = medical_risk_agent(input_data, condition)
        
    return results



test_user = {
    "age": 27,
    "height_cm": 155,
    "weight_kg": 74,
    "cycle_length_days": 40,
    "symptoms": ["acne", "irregular periods", "fatigue"],
    "family_history": ["diabetes"]
}

output = run_pipeline(test_user)
output





