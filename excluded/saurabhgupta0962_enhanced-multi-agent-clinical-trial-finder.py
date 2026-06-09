!pip install -U google-generativeai
!pip install google-generativeai --quiet


import pandas as pd
import logging
import time
from google import generativeai as genai
from kaggle_secrets import UserSecretsClient
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re
from IPython.display import display, Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("Gemini_LLM")
genai.configure(api_key=GOOGLE_API_KEY)


llm = genai.GenerativeModel("models/gemini-2.5-flash")

# Test the model
resp = llm.generate_content("Explain RNA-seq in 2 lines.")
print(resp.text)


df = pd.read_csv("/kaggle/input/covid19-clinical-trials-dataset/COVID clinical trials.csv")
df = df.dropna(subset=["Title", "Conditions"])
df_small = df.sample(300, random_state=42).reset_index(drop=True)

print(f"Loaded {len(df_small)} clinical trials")
df_small.head()


class MemoryStore:
    def __init__(self):
        self.store = {}
    
    def save(self, key, value):
        self.store[key] = value
    
    def get(self, key):
        return self.store.get(key)

memory = MemoryStore()


def safe_json(text):
    """
    Extracts valid JSON from an LLM response.
    If JSON cannot be parsed, returns a minimal safe dictionary.
    """
    if not text or not isinstance(text, str):
        return {"error": "empty response"}
    
    # Remove markdown code fences
    text = text.replace("```json", "").replace("```", "").strip()
    
    # Try direct JSON load
    try:
        return json.loads(text)
    except:
        pass
    
    # Try extracting JSON between braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Safe fallback
    return {"raw_text": text.strip(), "parsed": False}


class Agent:
    def __init__(self, name: str, description: str, model):
        self.name = name
        self.description = description
        self.model = model
    
    def run(self, state: dict) -> dict:
        raise NotImplementedError("Subclasses must implement run().")


class IntakeAgent(Agent):
    def run(self, state):
        logging.info("Running IntakeAgent")
        
        user_text = state["input_text"]
        
        prompt = f"""
Extract a structured patient profile from the following text.
Return JSON only with these fields:
- age (number)
- sex (string)
- condition (string)
- symptoms (list of strings)
- comorbidities (list of strings)
- location (string)

Text:
{user_text}
"""
        
        response = self.model.generate_content(prompt)
        profile_json = safe_json(response.text)
        
        state["patient_profile"] = profile_json
        memory.save("patient_profile", profile_json)
        
        return state


class RetrievalAgent(Agent):
    def run(self, state):
        logging.info("Running RetrievalAgent")
        
        patient = state.get("patient_profile", {})
        location = patient.get("location", "")
        condition = "COVID"
        
        # Filter by condition
        subset = df_small[df_small["Conditions"].str.contains(condition, case=False, na=False)]
        
        # Filter by location if available
        if location:
            location_filter = subset["Locations"].str.contains(location, case=False, na=False)
            location_trials = subset[location_filter]
            
            if len(location_trials) > 0:
                logging.info(f"Found {len(location_trials)} trials in {location}")
                subset = location_trials
            else:
                logging.info(f"No trials found in {location}, showing all trials")
        
        subset = subset.replace({np.nan: None})
        
        examples = (
            subset[["NCT Number", "Title", "Phases", "Status", "Locations"]]
            .head(10)
            .to_dict(orient="records")
        )
        
        tool_output = {
            "condition": condition,
            "location": location,
            "total_trials": len(subset),
            "location_specific": len(location_trials) if location else 0,
            "examples": examples
        }
        
        state["tool_result"] = tool_output
        memory.save("tool_result", tool_output)
        
        return state


class EligibilityAgent(Agent):
    def run(self, state):
        logging.info("Running EligibilityAgent")
        
        patient = state["patient_profile"]
        trials = state["tool_result"]["examples"]
        
        scored_trials = []
        
        for trial in trials[:5]:  # Score top 5 to save time
            prompt = f"""
Based on this patient profile:
Age: {patient.get('age', 'Unknown')}
Sex: {patient.get('sex', 'Unknown')}
Condition: {patient.get('condition', 'Unknown')}
Symptoms: {patient.get('symptoms', [])}
Comorbidities: {patient.get('comorbidities', [])}
Location: {patient.get('location', 'Unknown')}

Analyze this clinical trial:
Title: {trial['Title']}
NCT: {trial['NCT Number']}
Status: {trial['Status']}
Phase: {trial.get('Phases', 'Not specified')}
Location: {trial.get('Locations', 'Not specified')}

Provide a structured eligibility assessment.
Return ONLY valid JSON with this exact structure:
{{
  "score": 85,
  "matches": ["age appropriate", "location available"],
  "concerns": ["requires travel", "phase 1 trial"],
  "recommendation": "Highly Suitable"
}}

Recommendations must be one of: "Highly Suitable", "Suitable", "Review with Doctor", "Not Suitable"
Score must be 0-100.
"""
            
            try:
                response = self.model.generate_content(prompt)
                result = safe_json(response.text)
                
                # Validate and set defaults
                if not isinstance(result.get("score"), (int, float)):
                    result["score"] = 50
                if not isinstance(result.get("matches"), list):
                    result["matches"] = ["General COVID research"]
                if not isinstance(result.get("concerns"), list):
                    result["concerns"] = ["Consult doctor for details"]
                if "recommendation" not in result:
                    result["recommendation"] = "Review with Doctor"
                
                scored_trials.append({
                    "trial": trial,
                    "eligibility": result
                })
                
            except Exception as e:
                logging.error(f"Error scoring trial {trial['NCT Number']}: {e}")
                # Add with default score
                scored_trials.append({
                    "trial": trial,
                    "eligibility": {
                        "score": 50,
                        "matches": ["General COVID research"],
                        "concerns": ["Unable to assess - consult doctor"],
                        "recommendation": "Review with Doctor"
                    }
                })
        
        # Sort by score (highest first)
        scored_trials.sort(key=lambda x: x["eligibility"].get("score", 0), reverse=True)
        
        state["scored_trials"] = scored_trials
        memory.save("scored_trials", scored_trials)
        
        logging.info(f"Scored {len(scored_trials)} trials")
        
        return state


class ExplainerAgent(Agent):
    def run(self, state):
        logging.info("Running ExplainerAgent")
        
        tool_info = memory.get("tool_result")
        scored_trials = memory.get("scored_trials")
        
        prompt = f"""
Explain the following clinical trial information in simple, patient-friendly language.

Total trials found: {tool_info.get('total_trials', 0)}
Location-specific trials: {tool_info.get('location_specific', 0)}

Top scored trials:
{json.dumps([{
    "title": t["trial"]["Title"],
    "score": t["eligibility"]["score"],
    "recommendation": t["eligibility"]["recommendation"]
} for t in scored_trials[:3]], indent=2)}

Write a 2-3 paragraph explanation that:
1. Explains what clinical trials are
2. Describes why these trials were selected
3. Emphasizes the importance of consulting a doctor

Use simple language suitable for patients without medical background.
"""
        
        response = self.model.generate_content(prompt)
        explanation = response.text
        
        state["explanation"] = explanation
        memory.save("explanation", explanation)
        
        return state


def create_eligibility_chart(scored_trials):
    """Create a horizontal bar chart showing eligibility scores"""
    
    if not scored_trials:
        return None
    
    # Extract data
    trials = [t["trial"]["NCT Number"] for t in scored_trials[:5]]
    scores = [t["eligibility"].get("score", 0) for t in scored_trials[:5]]
    recommendations = [t["eligibility"].get("recommendation", "") for t in scored_trials[:5]]
    
    # Color coding
    colors = []
    for score in scores:
        if score >= 70:
            colors.append('#2ecc71')  # Green - Highly suitable
        elif score >= 50:
            colors.append('#f39c12')  # Orange - Suitable
        else:
            colors.append('#e74c3c')  # Red - Review needed
    
    # Create chart
    plt.figure(figsize=(12, 7))
    bars = plt.barh(trials, scores, color=colors)
    
    plt.xlabel('Eligibility Score (%)', fontsize=13, weight='bold')
    plt.ylabel('Clinical Trial (NCT Number)', fontsize=13, weight='bold')
    plt.title('ğŸ�¯ Top 5 Clinical Trials - Eligibility Match Score', 
              fontsize=15, weight='bold', pad=20)
    plt.xlim(0, 100)
    
    # Add score labels
    for i, (trial, score, rec) in enumerate(zip(trials, scores, recommendations)):
        plt.text(score + 2, i, f'{int(score)}%', 
                va='center', fontsize=11, weight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Highly Suitable (70-100)'),
        Patch(facecolor='#f39c12', label='Suitable (50-69)'),
        Patch(facecolor='#e74c3c', label='Review with Doctor (<50)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig('eligibility_scores.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    logging.info("Eligibility chart created: eligibility_scores.png")
    return 'eligibility_scores.png'


class ReportAgent(Agent):
    def run(self, state):
        logging.info("Running ReportAgent")
        
        patient = memory.get("patient_profile")
        tool_res = memory.get("tool_result")
        scored_trials = memory.get("scored_trials")
        explanation = memory.get("explanation")
        
        # Format patient profile
        try:
            patient_json = json.dumps(patient, indent=2)
        except:
            patient_json = "{}"
        
        # Build scored trials section
        trials_section = "## ğŸ�¯ Matched Clinical Trials (Ranked by Eligibility)\n\n"
        
        if scored_trials:
            for idx, item in enumerate(scored_trials, 1):
                trial = item["trial"]
                elig = item["eligibility"]
                
                score = elig.get("score", 0)
                recommendation = elig.get("recommendation", "Unknown")
                matches = elig.get("matches", [])
                concerns = elig.get("concerns", [])
                
                # Emoji based on score
                if score >= 70:
                    emoji = "âœ…"
                elif score >= 50:
                    emoji = "âš ï¸�"
                else:
                    emoji = "â�Œ"
                
                trials_section += f"""
### {emoji} Trial #{idx}: {trial['Title']}

| **Attribute** | **Details** |
|---------------|-------------|
| **NCT Number** | [{trial['NCT Number']}](https://clinicaltrials.gov/study/{trial['NCT Number']}) |
| **Eligibility Score** | **{score}/100** |
| **Recommendation** | **{recommendation}** |
| **Trial Status** | {trial['Status']} |
| **Phase** | {trial.get('Phases', 'Not specified')} |
| **Location** | {trial.get('Locations', 'Not specified')[:100]}... |

**âœ¨ Why This Trial Matches:**
{chr(10).join(['- ' + m for m in matches])}

**âš ï¸� Important Considerations:**
{chr(10).join(['- ' + c for c in concerns])}

**ğŸ”— Full Details:** Visit [ClinicalTrials.gov](https://clinicaltrials.gov/study/{trial['NCT Number']})

---
"""
        else:
            trials_section += "\nNo trials were scored. Please check the data.\n"
        
        # Create visualization
        chart_path = create_eligibility_chart(scored_trials)
        
        chart_section = ""
        if chart_path:
            chart_section = f"""
## ğŸ“Š Visual Eligibility Analysis

![Eligibility Scores Chart]({chart_path})

*Chart shows eligibility match scores for top 5 trials. Higher scores indicate better match with patient profile.*

---
"""
        
        # Action items section
        action_items = """
## âœ… Next Steps - What To Do Now

### Immediate Actions:
1. **ğŸ“„ Print this report** and bring it to your next doctor's appointment
2. **ğŸ”� Review top-ranked trials** - Focus on those marked âœ… (70+ score)
3. **ğŸ‘¨â€�âš•ï¸� Consult your physician** - They can assess if you meet full eligibility criteria
4. **ğŸ“� Contact trial coordinators** - Use the NCT numbers to find contact information

### How to Find More Information:

Visit: `https://clinicaltrials.gov/study/[NCT_NUMBER]`

Example: For NCT04376944, visit:
https://clinicaltrials.gov/study/NCT04376944

Look for the **"Contacts and Locations"** section for:
- Principal investigator contact
- Study coordinator phone/email
- Trial site locations
- Recruitment status

### Questions to Ask Your Doctor:

1. **Am I truly eligible for these trials?**
   - Do I meet all inclusion criteria?
   - Are there any exclusion factors?

2. **What are the risks vs benefits?**
   - What are potential side effects?
   - What are expected benefits?

3. **How will this affect my current treatment?**
   - Can I continue my current medications?
   - Will this interfere with other treatments?

4. **What is the time commitment?**
   - How many visits are required?
   - How long is the trial period?
   - What follow-up is needed?

5. **What about costs?**
   - Is the trial treatment free?
   - Are travel costs covered?
   - What about insurance?

### ğŸ”� Understanding Trial Phases:

- **Phase 1**: Early safety testing, small group (20-80 people)
- **Phase 2**: Efficacy testing, medium group (100-300 people)
- **Phase 3**: Large-scale testing, confirmation (1,000-3,000 people)
- **Phase 4**: Post-market surveillance, long-term effects

**âš ï¸� Note:** Earlier phases (1-2) are more experimental and may carry higher risks.

---
"""
        
        # Build final report
        final_report = f"""
# ğŸ�¥ Clinical Trials Navigator - Personalized Report
### AI-Powered Trial Matching System

---

## ğŸ‘¤ Patient Profile
```json
{patient_json}
```

---

## ğŸ“ˆ Search Results Summary

- **Total COVID-19 Trials in Database:** {tool_res.get('total_trials', 0)}
- **Location-Specific Trials:** {tool_res.get('location_specific', 0)} in {tool_res.get('location', 'unspecified location')}
- **Trials Analyzed:** {len(scored_trials)}
- **Search Condition:** {tool_res.get('condition', 'COVID-19')}

---

{trials_section}

{chart_section}

## ğŸ“‹ Understanding Your Results

{explanation}

---

{action_items}

## âš ï¸� IMPORTANT MEDICAL DISCLAIMER

**This report is generated by an AI-powered system and is NOT a substitute for professional medical advice.**

- âœ‹ **Do NOT enroll in any trial without consulting your healthcare provider**
- ğŸ‘¨â€�âš•ï¸� **Your doctor must review your full medical history and current health status**
- ğŸ“‹ **Each trial has specific eligibility criteria that require medical evaluation**
- ğŸ”¬ **Clinical trials involve experimental treatments with potential risks**
- ğŸ“� **Always verify trial information directly with ClinicalTrials.gov**

### Your Health, Your Choice

Clinical trials are voluntary. You have the right to:
- Ask questions before deciding
- Take time to make your decision
- Withdraw at any time
- Get a second opinion

---

## ğŸ“š Additional Resources

- **ClinicalTrials.gov**: https://clinicaltrials.gov
- **FDA Clinical Trials Information**: https://www.fda.gov/patients/clinical-trials-what-patients-need-know
- **WHO Clinical Trials Registry**: https://www.who.int/clinical-trials-registry-platform

---

*Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*
*Powered by Gemini 2.5 Flash AI*
"""
        
        state["final_report"] = final_report
        memory.save("final_report", final_report)
        
        return state


def run_pipeline(user_text):
    """
    Execute the complete multi-agent pipeline
    """
    state = {"input_text": user_text}
    
    agents = [
        IntakeAgent("intake", "Extract patient profile", llm),
        RetrievalAgent("retrieval", "Search clinical trials database", llm),
        EligibilityAgent("eligibility", "Score trial matches", llm),  # â­� NEW
        ExplainerAgent("explainer", "Generate patient-friendly explanation", llm),
        ReportAgent("report", "Create comprehensive report", llm)
    ]
    
    logging.info("=" * 60)
    logging.info("STARTING CLINICAL TRIALS NAVIGATOR PIPELINE")
    logging.info("=" * 60)
    
    for agent in agents:
        t1 = time.time()
        state = agent.run(state)
        t2 = time.time()
        logging.info(f"âœ“ {agent.name.upper()} completed in {t2 - t1:.2f}s")
    
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)
    
    return state


print("="*80)
print("TEST CASE 1: Brazilian Patient with Long COVID")
print("="*80)

user_text_1 = """
My father is 62 years old, male, diabetic, with severe breathing problems after COVID-19.
He had COVID 6 months ago and still has shortness of breath and fatigue.
He takes metformin for diabetes.
We live in SÃ£o Paulo, Brazil.
We want to know if there are any COVID-19 clinical trials that might help his recovery.
"""

result_1 = run_pipeline(user_text_1)
print("\n")
print(result_1["final_report"])

# Display chart if created
from IPython.display import Image, display
try:
    display(Image('eligibility_scores.png'))
except:
    print("Chart not found")


print("\n\n")
print("="*80)
print("TEST CASE 2: US Patient with Comorbidities")
print("="*80)

user_text_2 = """
I am a 55-year-old female with asthma and high blood pressure.
I have recovered from COVID-19, but I would like to participate in vaccine trials or prevention studies.
I live in New York, USA.
Looking for Phase 3 trials, if possible.
"""

result_2 = run_pipeline(user_text_2)
print("\n")
print(result_2["final_report"])


def evaluate_agent_performance(state):
    """
    Evaluate the quality of agent outputs
    """
    
    metrics = {
        "patient_profile_extracted": isinstance(state.get("patient_profile"), dict),
        "trials_retrieved": len(state.get("tool_result", {}).get("examples", [])) > 0,
        "eligibility_scored": len(state.get("scored_trials", [])) > 0,
        "explanation_generated": len(state.get("explanation", "")) > 50,
        "report_created": len(state.get("final_report", "")) > 100,
        "chart_created": state.get("chart_path") is not None
    }
    
    score = sum(metrics.values()) / len(metrics) * 100
    
    print("\n" + "="*60)
    print("AGENT PERFORMANCE EVALUATION")
    print("="*60)
    
    for metric, passed in metrics.items():
        status = "âœ… PASS" if passed else "â�Œ FAIL"
        print(f"{status} - {metric}")
    
    print(f"\nOverall Score: {score:.1f}%")
    print("="*60)
    
    return metrics, score

# Run evaluation
metrics, score = evaluate_agent_performance(result_1)


def export_results(state, filename="clinical_trials_results.json"):
    """
    Export complete results to JSON file
    """
    
    export_data = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "patient_profile": state.get("patient_profile"),
        "search_results": state.get("tool_result"),
        "scored_trials": state.get("scored_trials"),
        "explanation": state.get("explanation"),
        "chart_path": state.get("chart_path")
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"âœ… Results exported to: {filename}")
    return filename

# Export results
export_results(result_1)

