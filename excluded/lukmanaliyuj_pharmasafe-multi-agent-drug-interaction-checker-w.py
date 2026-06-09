import os
from kaggle_secrets import UserSecretsClient
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


"""
PharmaSafe: Multi-Agent Drug Interaction Checker
A sophisticated drug interaction analysis system using Google ADK with Gemini 2.5 flash lite

This project demonstrates:
- Multi-agent architecture with specialized agents
- Sequential workflow for structured analysis
- Integration of Cohere's Command models via LiteLLM with ADK
- Pharmaceutical safety checking with AI

Course: Kaggle 5-Day Agents Intensive
Framework: Google Agent Development Kit (ADK)
LLM Provider: Cohere (via LiteLLM)
"""

import os
from google.adk.agents import Agent, SequentialAgent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# Configure retry options for robustness
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("=" * 70)
print("ğŸ’Š PharmaSafe: Multi-Agent Drug Interaction Checker")
print("=" * 70)
print("\nğŸ”§ System Configuration:")
print("   - Framework: Google ADK")
print("   - LLM Provider: Cohere Command-A-03")
print("   - Architecture: 4-Agent Sequential Pipeline")
print("\n" + "=" * 70)

gemini_llm = Gemini(model='gemini-2.5-flash-lite')
# ============================================================================
# AGENT 1: MEDICATION PARSER
# ============================================================================
medication_parser = LlmAgent(
    model=gemini_llm, 
    name="MedicationParser",
    instruction="""You are a medication parsing specialist. Your task is to:

1. Extract all medications mentioned by the user
2. Identify dosages if provided
3. Note timing information (morning, evening, with food, etc.)
4. List any supplements or herbal products
5. Organize the information clearly

Format your output as:
**Identified Medications:**
- [Medication 1]: [dosage if known]
- [Medication 2]: [dosage if known]

**Supplements/Herbals:**
- [List any supplements]

**Timing Information:**
- [Any timing details mentioned]

**User Questions/Concerns:**
- [Any specific questions asked]

Be thorough and precise. Extract every medication detail.""",
    output_key="parsed_medications"
)

print("âœ… Agent 1: MedicationParser - Ready")

# ============================================================================
# AGENT 2: INTERACTION ANALYZER
# ============================================================================
interaction_analyzer = LlmAgent(
    model=gemini_llm,
    name="InteractionAnalyzer",
    instruction="""You are a clinical pharmacology expert specializing in drug interactions.

Review the parsed medications: {parsed_medications}

Analyze and identify ALL potential interactions:

1. **ğŸ”´ MAJOR INTERACTIONS** (potentially life-threatening)
   - List drug pairs with major interactions
   - Explain the pharmacological mechanism
   - Describe potential serious consequences
   - Note contraindications

2. **ğŸŸ¡ MODERATE INTERACTIONS** (may require monitoring)
   - List drug pairs with moderate interactions
   - Explain clinical significance
   - Specify monitoring requirements
   - Note dose adjustments if needed

3. **ğŸŸ¢ MINOR INTERACTIONS** (generally manageable)
   - List drug pairs with minor interactions
   - Provide management tips
   - Note if action is needed

4. **ğŸ�½ï¸� FOOD INTERACTIONS**
   - Identify any food-drug interactions
   - Provide specific dietary guidance
   - Note timing with meals

5. **â�° TIMING CONSIDERATIONS**
   - Recommend optimal spacing between medications
   - Note medications that should be taken together
   - Identify medications to separate

Use clear medical terminology but explain concepts accessibly.
Be comprehensive and evidence-based in your analysis.""",
    output_key="interaction_analysis"
)

print("âœ… Agent 2: InteractionAnalyzer - Ready")

# ============================================================================
# AGENT 3: SAFETY ADVISOR
# ============================================================================
safety_advisor = LlmAgent(
    model=gemini_llm,
    name="SafetyAdvisor",
    instruction="""You are a patient safety specialist and clinical pharmacist.

Based on the interaction analysis: {interaction_analysis}

Provide comprehensive, actionable safety recommendations:

1. **âš ï¸� IMMEDIATE ACTIONS REQUIRED**
   - List any urgent concerns requiring immediate medical attention
   - Highlight medications that should not be taken together
   - Specify if emergency consultation is needed

2. **ğŸ“‹ MONITORING RECOMMENDATIONS**
   - Specify symptoms to watch for
   - Suggest laboratory tests if applicable (with ranges)
   - Provide monitoring timeline
   - Note frequency of follow-up

3. **â�° DOSING & TIMING ADJUSTMENTS**
   - Recommend specific timing for each medication
   - Suggest spacing between interacting drugs
   - Note any dose modifications (emphasize pharmacist approval needed)
   - Provide a sample daily medication schedule

4. **ğŸ“š PATIENT EDUCATION POINTS**
   - Key information the patient must know
   - Common side effects to expect
   - Warning signs that require immediate attention
   - Questions to ask their pharmacist/doctor

5. **âœ… NEXT STEPS**
   - Prioritized action items
   - Timeline for each action
   - Who to contact (pharmacist, doctor, emergency)

Be specific, practical, and always emphasize professional consultation.""",
    output_key="safety_recommendations"
)

print("âœ… Agent 3: SafetyAdvisor - Ready")

# ============================================================================
# AGENT 4: REPORT FORMATTER
# ============================================================================
report_formatter = LlmAgent(
    model=gemini_llm,
    name="ReportFormatter",
    instruction="""You are a medical communication specialist creating patient-friendly reports.

Compile a comprehensive, well-formatted report using:
- Parsed medications: {parsed_medications}
- Interaction analysis: {interaction_analysis}
- Safety recommendations: {safety_recommendations}

Create a clear, actionable report with this EXACT structure:

# ğŸ’Š PharmaSafe Drug Interaction Report

## ğŸ“� Your Medications Summary
[List all identified medications with dosages clearly]

## âš ï¸� Interaction Analysis

### ğŸ”´ Major Interactions (Urgent Attention Required)
[List major interactions or state "None identified"]

### ğŸŸ¡ Moderate Interactions (Monitoring Needed)
[List moderate interactions or state "None identified"]

### ğŸŸ¢ Minor Interactions
[List minor interactions or state "None identified"]

### ğŸ�½ï¸� Food & Lifestyle Interactions
[List food/alcohol interactions or state "None identified"]

## ğŸ›¡ï¸� Safety Recommendations

### âš ï¸� Immediate Actions
[List urgent actions or state "No immediate actions required"]

### ğŸ“‹ Monitoring Plan
[Provide monitoring guidance]

### â�° Medication Timing Schedule
[Provide a sample daily schedule]

## âœ… Your Action Plan

1. [First priority action]
2. [Second priority action]
3. [Additional actions...]

## ğŸ“� When to Contact Healthcare Providers

**Contact your pharmacist within 24 hours if:**
- [Specific situations]

**Contact your doctor if:**
- [Specific situations]

**Seek emergency help immediately if:**
- [Emergency warning signs]

---

## âš ï¸� IMPORTANT MEDICAL DISCLAIMER

**This analysis is for educational purposes only and does not replace professional medical advice.**

âœ‹ **DO NOT:**
- Stop or change medications without consulting your healthcare provider
- Start new medications without professional guidance
- Ignore symptoms or side effects

âœ… **DO:**
- Share this report with your pharmacist or doctor
- Ask questions about your medications
- Report any unusual symptoms immediately
- Keep all follow-up appointments

**Critical Reminders:**
- Drug interactions can be complex and patient-specific
- Your medical history, allergies, and other conditions matter
- Only a licensed healthcare provider can make medication decisions
- This AI analysis may not capture all relevant factors

---

*Report generated by PharmaSafe AI Multi-Agent System*
*Powered by: Google ADK + Gemini*
*Date: 25-Nov-2025*

Make the report professional, compassionate, evidence-based, and immediately actionable.""",
    output_key="final_report"
)

print("âœ… Agent 4: ReportFormatter - Ready")

# ============================================================================
# CREATE SEQUENTIAL PIPELINE
# ============================================================================
pharmasafe_pipeline = SequentialAgent(
    name="PharmaSafePipeline",
    sub_agents=[
        medication_parser,
        interaction_analyzer,
        safety_advisor,
        report_formatter
    ]
)

print("âœ… Sequential Pipeline: PharmaSafePipeline - Assembled")
print("\n" + "=" * 70)
print("ğŸ�¯ Multi-Agent System Ready!")
print("=" * 70 + "\n")


# ============================================================================
# MAIN INTERACTION FUNCTION
# ============================================================================
async def check_drug_interactions(user_query: str) -> str:
    """
    Main function to check drug interactions using the multi-agent pipeline
    
    Args:
        user_query: User's medication information and questions
        
    Returns:
        Comprehensive drug interaction report
        
    Example:
        result = await check_drug_interactions(
            "I'm taking warfarin and ibuprofen. Are there interactions?"
        )
    """
    # Input validation
    if not user_query or len(user_query.strip()) < 5:
        return """
âš ï¸� **Insufficient Information**

Please provide:
- Names of medications you're taking
- Dosages (if known)
- Your specific questions or concerns

Example: "I'm taking warfarin 5mg daily and ibuprofen 400mg as needed. Are there any interactions?"
        """
    
    try:
        print(f"\n{'='*70}")
        print("ğŸ”„ Processing your query through the multi-agent pipeline...")
        print(f"{'='*70}\n")
        
        # Run the sequential pipeline
        runner = InMemoryRunner(agent=pharmasafe_pipeline)
        response = await runner.run_debug(user_query)
        
        print(f"\n{'='*70}")
        print("âœ… Analysis complete!")
        print(f"{'='*70}\n")
        
        return response
        
    except Exception as e:
        error_report = f"""
â�Œ **Error Processing Request**

An error occurred while analyzing your medications: {str(e)}

**What to do:**
1. Verify your COHERE_API_KEY is set correctly
2. Check your internet connection
3. Try rephrasing your query
4. If the issue persists, contact support

**In the meantime:**
- Contact your pharmacist directly for urgent concerns
- Call your doctor if you're experiencing symptoms
- Don't stop or change medications without guidance
        """
        return error_report


# ============================================================================
# EXAMPLE QUERIES FOR DEMONSTRATION
# ============================================================================
example_queries = {
    "anticoagulant_nsaid": "I'm taking warfarin 5mg daily and my doctor prescribed ibuprofen 400mg three times daily for pain. Are there any interactions?",
    
    "ppi_antiplatelet": "Can I take omeprazole 20mg once daily with clopidogrel 75mg daily? I have both prescribed.",
    
    "diabetes_alcohol": "I'm on metformin 500mg twice daily for diabetes and want to know if it's safe to drink alcohol occasionally.",
    
    "ace_potassium": "Taking lisinopril 10mg daily and considering potassium supplements for leg cramps. Any concerns?",
    
    "statin_antibiotic": "I have simvastatin 20mg at bedtime and just got prescribed clarithromycin 500mg twice daily for 10 days.",
    
    "multiple_meds": "I take: atorvastatin 40mg nightly, amlodipine 5mg daily, aspirin 81mg daily, and metformin 1000mg twice daily. Just prescribed azithromycin. Safe?",
}


# ============================================================================
# HELPER FUNCTION TO DISPLAY EXAMPLES
# ============================================================================
def show_examples():
    """Display example queries for users"""
    print("\n" + "="*70)
    print("ğŸ’¡ EXAMPLE QUERIES")
    print("="*70 + "\n")
    
    for i, (key, query) in enumerate(example_queries.items(), 1):
        print(f"{i}. **{key.replace('_', ' ').title()}:**")
        print(f"   '{query}'\n")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    print("\n" + "ğŸ�“ " + "="*68 + " ğŸ�“")
    print("   Kaggle 5-Day Agents Intensive - Final Project")
    print("   PharmaSafe: AI-Powered Drug Interaction Checker")
    print("ğŸ�“ " + "="*68 + " ğŸ�“\n")
    
    print("ğŸ“Š Architecture Overview:")
    print("   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�")
    print("   â”‚  User Query     â”‚")
    print("   â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜")
    print("            â”‚")
    print("   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�")
    print("   â”‚ 1. Medication Parser    â”‚ â†� Extracts medication info")
    print("   â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")
    print("            â”‚ parsed_medications")
    print("   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�")
    print("   â”‚ 2. Interaction Analyzer â”‚ â†� Identifies interactions")
    print("   â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")
    print("            â”‚ interaction_analysis")
    print("   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�")
    print("   â”‚ 3. Safety Advisor       â”‚ â†� Provides recommendations")
    print("   â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")
    print("            â”‚ safety_recommendations")
    print("   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�")
    print("   â”‚ 4. Report Formatter     â”‚ â†� Creates final report")
    print("   â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")
    print("            â”‚ final_report")
    print("   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�")
    print("   â”‚   User Receives Report  â”‚")
    print("   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜\n")
    
    # Show example queries
    show_examples()
    
    # Example execution
    print("ğŸ§ª Running Example Test Case...\n")
    test_query = example_queries["anticoagulant_nsaid"]
    print(f"Query: '{test_query}'\n")
    
    
    print("\nğŸ’» To run this system in your notebook:")
    print("   1. Ensure GOOGLE_API_KEY is set in environment")
    print("   2. Install: pip install google-adk")
    print("   3. Execute:")
    print("      import asyncio")
    print("      result = await check_drug_interactions(test_query)")
    print("      print(result)")
    print("\n" + "="*70)


# ğŸ§ª Running Example Test Case...

Query = "I'm taking rabeprazole 20mg daily and my doctor prescribed ibuprofen 400mg three times daily for pain. Are there any interactions?"

import asyncio
result = await check_drug_interactions(Query)
print(result)




