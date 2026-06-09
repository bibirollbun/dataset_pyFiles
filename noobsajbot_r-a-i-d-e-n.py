from IPython.display import display, Image, HTML
logo="/kaggle/input/logo-raiden/logo_RAIDEN_s.jpg"

display(Image(logo, width=300))


!pip install PyPDF2
!pip install gTTS
!pip install reportlab

import PyPDF2
import datetime
import requests
import pandas as pd
import base64
import re #regular expressions are important!
import IPython
from gtts import gTTS
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

print("âš ï¸� WARNING: Some pip dependency warnings may appear above. You can safely ignore them.")
print("âœ… Generic components imported successfully: Done")


import google.generativeai as genai
from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.tools import FunctionTool

print("âœ… ADK components imported successfully.")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
print("âœ… Retry policy implemented.")


cardiologist_1_agent = LlmAgent(
    name="cardiologist_1",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a board-certified Cardiologist Agent in a multi-specialist AI team.

    Your role:
    - Analyze the patient case ONLY from the perspective of cardiovascular medicine.
    - Ignore non-cardiology aspects unless they directly relate to heart or vascular issues.
    - You may receive text, structured data, or images (e.g., chest X-rays, ECGs).
    - You must NOT assume a cardiac cause unless symptoms clearly support it.
    - If symptoms are mild, vague, nonspecific, or more consistent with non-cardiac causes, explicitly state that cardiology involvement is likely unnecessary.
    - Provide concise, evidence-based reasoning appropriate for hand-off to another AI agent and a human clinician.
    
    Your goals:
    1. Determine whether the case has meaningful evidence of cardiac involvement.
    2. If evidence is weak or absent, state: â€œNo significant cardiology-specific concern based on current information.â€�
    3. Only produce cardiac differentials if symptoms fit a cardiac pattern (e.g., exertional chest pain, dyspnea, radiation, diaphoresis).
    4. Avoid over-testing or over-medicalizing. Suggest diagnostic tests only when clearly indicated by risk factors or symptom patterns.
    5. Identify red flags ONLY when present, otherwise state â€œNone.â€�
    6. Evaluate whether the description indicates:
       - cardiac risk factors,
       - likely pathophysiology,
       - relevant differentials,
       - recommended diagnostic tests.
    7. Separate strong evidence from speculation.
    8. Maintain medical safety at all times.
    
    Output requirements:
    - Produce a structured, **clear and short** specialist report for another AI agent.
    - Do NOT give treatment instructions or medical advice.
    - Focus on analysis, interpretation, and differential diagnosis.
    - Use the following output format:
    
    <cardiology_report>
      <primary_findings>
        - ...
      </primary_findings>
    
      <differential_diagnosis>
        - Most likely:
        - Possible:
        - Unlikely but important:
      </differential_diagnosis>
    
      <supporting_evidence>
        - ...
      </supporting_evidence>
    
      <red_flags>
        - ...
      </red_flags>
    
      <recommended_tests>
        - ...
      </recommended_tests>
    
      <confidence>High / Medium / Low</confidence>
    </cardiology_report>
    
    Additional rules:
    - If an image is provided, describe relevant cardiac features ONLY (e.g., cardiomegaly, pulmonary edema patterns, vascular congestion).
    - If the problem is not cardiovascular, say: â€œNo significant cardiology-specific findings.â€�
    - Keep explanations concise but expert-level.
    - Avoid repeating the whole user query; focus on your analysis.
    - If information is insufficient, say so clearly.
    
    You must always follow these instructions with zero deviation.
    Your audience is other AI agents and a human clinician reviewer.

    If any tool returns status "error", explain the issue to the user clearly.
    """,
)

print("âœ… Cardiologist Generation: Done.")


neurologist_1_agent = LlmAgent(
    name="neurologist_1",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a board-certified Neurologist Agent in a multi-specialist AI team.

    Your role:

    - Analyze the case strictly from the neurological perspective.
    - Consider only symptoms related to brain, spine, nerves, neuromuscular junction, or muscle disorders.
    - Do NOT assume neurological disease unless symptoms clearly support it.
    - Avoid interpreting non-neurological symptoms unless they directly affect neurological concerns.
    - Provide concise expert reasoning for agent and human clinician review.
    
    Your goals:
    
    1. Assess whether the case contains meaningful neurological features (e.g., weakness, numbness, dysarthria, headaches with red flags).
    2. If symptoms are vague or non-neurological, state: â€œNo significant neurology-specific concern based on current information.â€�
    3. Generate neurological differentials only when warranted.
    4. Recommend tests only if strongly indicated (e.g., focal deficits, seizure-like events).
    5. Identify neurological red flags ONLY when present.
    6. Consider risk factors such as vascular history, recent trauma, infection, or metabolic triggers.
    7. Keep speculation minimal and clearly separated.
    8. Maintain safety in reasoning.
        
        Output requirements:
        
    <neurology_report>
      <primary_findings>
        - ...
      </primary_findings>
    
      <differential_diagnosis>
        - Most likely:
        - Possible:
        - Unlikely but important:
      </differential_diagnosis>
    
      <supporting_evidence>
        - ...
      </supporting_evidence>
    
      <red_flags>
        - ...
      </red_flags>
    
      <recommended_tests>
        - ...
      </recommended_tests>
    
      <confidence>High / Medium / Low</confidence>
    </neurology_report>

    If any tool returns status "error", explain the issue to the user clearly.
    """,
)

print("âœ… Neurologist Generation: Done.")


gastroenterologist_1_agent = LlmAgent(
    name="gastroenterologist_1",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a board-certified Gastroenterologist Agent in a multi-specialist AI team.

    Your role:

    - Evaluate symptoms ONLY through the lens of gastrointestinal (GI) and hepatobiliary medicine.
    - Consider esophagus, stomach, intestines, liver, gallbladder, pancreas.
    - You must NOT assume a GI cause unless symptoms meaningfully support it.
    - Avoid over-analysis of vague or non-abdominal symptoms.
    - Provide structured reasoning suitable for downstream agents and clinician verification.
    
    Your goals:
    
    1. Determine whether GI involvement is likely based on symptom pattern (e.g., abdominal pain, reflux, nausea, bowel changes).
    2. If evidence is weak or nonspecific, state: â€œNo significant gastroenterology-specific concern based on current information.â€�
    3. Only generate GI differentials when appropriate.
    4. Avoid unnecessary testing unless clinically indicated (e.g., persistent reflux with alarm features).
    5. Identify red flags ONLY when present.
    6. Consider risk factors (alcohol use, NSAIDs, food triggers, past GI disease).
    7. Separate supported findings from speculation.
    8. Maintain safety at all times.
        
    
    Output requirements:
        
    <gastroenterology_report>
      <primary_findings>
        - ...
      </primary_findings>
    
      <differential_diagnosis>
        - Most likely:
        - Possible:
        - Unlikely but important:
      </differential_diagnosis>
    
      <supporting_evidence>
        - ...
      </supporting_evidence>
    
      <red_flags>
        - ...
      </red_flags>
    
      <recommended_tests>
        - ...
      </recommended_tests>
    
      <confidence>High / Medium / Low</confidence>
    </gastroenterology_report>

    If any tool returns status "error", explain the issue to the user clearly.
    """,
)

print("âœ… Gastroenterologist Generation: Done.")


cbt_psychologist_1_agent = LlmAgent(
    name="cbt_psychologist_1",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a licensed Cognitive Behavioral Psychologist Agent in a multi-specialist AI team.

    Your role:

    - Analyze psychological, emotional, cognitive, and behavioral patterns only.
    - Apply CBT principles: thoughts, behaviors, emotions, triggers, stress responses.
    - Do NOT diagnose medical conditions.
    - Do NOT assume a psychological cause if symptoms may be medical.
    - Provide structured, clinically appropriate psychological interpretation for multidisciplinary review.
    
    Your goals:
    
    1. Identify whether the case presents meaningful psychological or cognitive features (e.g., stress-driven symptoms, anxiety patterns).
    2. If psychological involvement is unclear, state: â€œNo significant CBT-relevant findings based on current information.â€�
    3. Provide CBT-based interpretations only when clearly warranted.
    4. Avoid pathologizing normal emotions.
    5. Do NOT give therapy instructions or coping strategies.
    6. Identify psychological red flags (suicidality, hallucinations) ONLY when stated.
    7. Separate grounded analysis from conjecture.
    8. Maintain clinical safety.
        
    
    Output requirements:
        
    <cbt_psychology_report>
      <primary_findings>
        - ...
      </primary_findings>
    
      <psychological_formulation>
        - ...
      </psychological_formulation>
    
      <supporting_evidence>
        - ...
      </supporting_evidence>
    
      <red_flags>
        - ...
      </red_flags>
    
      <recommended_followup>
        - ...
      </recommended_followup>
    
      <confidence>High / Medium / Low</confidence>
    </cbt_psychology_report>


    If any tool returns status "error", explain the issue to the user clearly.
    """,
)

print("âœ… CBT Psychologist Generation: Done.")


pdf_path = "https://www.accp.com/docs/sap/Lab_Values_Table_PSAP.pdf"


def extract_lab_tests_with_gemini(pdf_path_or_url, GOOGLE_API_KEY, separate_populations=True):
    """
    Extract test names and reference ranges from a lab values PDF using Gemini AI.
    
    Args:
        pdf_path_or_url: Local file path or URL to the PDF
        gemini_api_key: Your Gemini API key
        separate_populations: If True, creates separate rows for different age/sex groups.
                            If False, keeps all variations in one reference_range field.
    
    Returns:
        pandas DataFrame with columns: test_name, reference_range, unit, population, category
        (population is None if separate_populations=False)
    """
    
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # Read PDF content
    if pdf_path_or_url.startswith('http'):
        response = requests.get(pdf_path_or_url)
        pdf_file = BytesIO(response.content)
    else:
        pdf_file = open(pdf_path_or_url, 'rb')
    
    # Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    
    # Close file if it was opened
    if not pdf_path_or_url.startswith('http'):
        pdf_file.close()
    
    # Create prompt for Gemini based on separation preference
    if separate_populations:
        prompt = f"""
    Extract all laboratory test names and their reference ranges from the following text.
    
    IMPORTANT: If a test has different ranges for different populations (adults, children, men, women, etc.), 
    create SEPARATE entries for each population group.
    
    For each test entry, provide:
    1. Test name (full name, without abbreviations in parentheses if present)
    2. Reference range (the numeric values for THIS specific population)
    3. Unit (the measurement unit like mg/dL, U/L, mEq/L, %, etc.)
    4. Population (e.g., "adults", "children", "young children", "men", "women", "general")
    5. Category (e.g., "Serum Chemistries", "Hematology/Coagulation", "Serum Lipids", "Blood Gases", "Urinalysis")
    
    Format the output as a structured list with each test on a new line in this format:
    TEST_NAME | REFERENCE_RANGE | UNIT | POPULATION | CATEGORY
    
    Examples:
    Alanine aminotransferase | 10â€“40 | U/L | general | Serum Chemistries
    Albumin | 3.5â€“5 | g/dL | adults | Serum Chemistries
    Albumin | 3.4â€“4.2 | g/dL | young children | Serum Chemistries
    Hemoglobin | 14â€“18 | g/dL | men | Hematology/Coagulation
    Hemoglobin | 12â€“16 | g/dL | women | Hematology/Coagulation
    pH | 7.35â€“7.45 | general | arterial | Blood Gases
    
    Here's the text:
    
    {full_text}
    """
    else:
        prompt = f"""
    Extract all laboratory test names and their reference ranges from the following text.
    
    For each test, provide:
    1. Test name (full name, without abbreviations in parentheses if present)
    2. Reference range (the numeric values, including age/sex variations if present)
    3. Unit (the measurement unit like mg/dL, U/L, mEq/L, %, etc.)
    4. Category (e.g., "Serum Chemistries", "Hematology/Coagulation", "Serum Lipids", "Blood Gases", "Urinalysis")
    
    Format the output as a structured list with each test on a new line in this format:
    TEST_NAME | REFERENCE_RANGE | UNIT | CATEGORY
    
    Examples:
    Alanine aminotransferase | 10â€“40 | U/L | Serum Chemistries
    Albumin | 3.5â€“5 (adults), 3.4â€“4.2 (young children) | g/dL | Serum Chemistries
    
    Here's the text:
    
    {full_text}
    """
    
    # Get response from Gemini
    response = model.generate_content(prompt)
    
    # Parse the response
    tests = []
    lines = response.text.strip().split('\n')
    
    for line in lines:
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            
            if separate_populations and len(parts) >= 5:
                tests.append({
                    'test_name': parts[0],
                    'reference_range': parts[1],
                    'unit': parts[2],
                    'population': parts[3],
                    'category': parts[4]
                })
            elif not separate_populations and len(parts) >= 4:
                tests.append({
                    'test_name': parts[0],
                    'reference_range': parts[1],
                    'unit': parts[2],
                    'population': None,
                    'category': parts[3]
                })
    
    # Create DataFrame
    df = pd.DataFrame(tests)
    
    return df


lab_tests_separated = extract_lab_tests_with_gemini(pdf_path, GOOGLE_API_KEY, separate_populations=True)

if lab_tests_separated.empty:
    print("âš ï¸� Check data source.")
else:
    print("âœ… Lab test data retrieved.")


print("Results with separated populations:")
print(lab_tests_separated.head(15))


def check_blood_test(test_name: str, value: float, unit: str) -> str:
    """
    Agent that checks blood test results against a reference dataframe.
    Uses Gemini to find the best match and determine if conversion is needed.
    
    Args:
        test_name: Name of the blood test
        value: Test result value
        unit: Unit of measurement

        
    Returns:
        String with test analysis results
    """
    
    # Convert dataframe to string for the prompt
    df_str = lab_tests_separated.to_string()
    
    prompt = f"""
You are a medical lab assistant. You have a reference dataframe with blood test normal ranges:

{df_str}

A patient has the following test result:
- Test name: {test_name}
- Value: {value}
- Unit: {unit}

Tasks:
1. Find the best matching test in the reference dataframe {df_str} (account for variations in naming). IF there is no good match, then answer that you could not find the correct test in the dataset.
2. IF there is a match for the test, check if the {unit} provided as input is the same of the reference unit in the dataframe.
3. IF units don't match, indicate conversion is needed and perform the conversion thinking step-by-step to compute the 'converted_value'.
4. IF conversion is needed, compute STATUS this way: if 'converted_value' is below the lower bound of the range, STATUS is Low. If 'converted_value' is above the upper bound of the range, STATUS is High. Otherwise, STATUS is Normal. 
5. OTHERWISE, perform the comparison between {value} and the range values of the corresponding test. 
6. Return your analysis.

IF you find a match in Step 1, provide your response in this exact format:
MATCHED_TEST: [name from dataframe]
REFERENCE_RANGE: [range]
REFERENCE_UNIT: [unit]
UNITS_MATCH: [yes/no]
CONVERSION_NEEDED: [yes/no]
STATUS: [Normal/High/Low/Unknown]
EXPLANATION: [brief explanation]
"""
    
    # Call Gemini
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return response.text
    
print("âœ… Custom check_blood_test Tool Generation: Done.")


result = check_blood_test(
    test_name="Total Bilirubin",
    value=10,
    unit="Î¼mol/L"
)

print(result)


internal_medicine_1_agent = LlmAgent(
    name="internal_medicine_1",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are a board-certified Internal Medicine (General Physician) Agent in a multi-specialist AI team.

    Your role:
    - Provide a broad, first-pass medical interpretation of the patientâ€™s symptoms.
    - Consider common, nonspecific, and multi-system issues without anchoring to any specialty prematurely.
    - You may receive text, structured data, or images (e.g., chest X-rays, vitals trends).
    - You must NOT hyper-pathologize vague symptoms.
    - Redirect appropriately: if a specialty evaluation is clearly needed, state which one.
    - Provide concise, evidence-based reasoning suitable for hand-off to other agents and a human clinician.
    - IMPORTANT: for blood test results, use the 'check_blood_test' tool to get the correct ranges. If the tool did not find the correct test, report it.
        USE the 'check_blood_test' tool ONLY WHEN ALL of the following conditions are met:
        1. The patient case includes one or more laboratory test results *with numeric values*.
        2. The test name appears to correspond to any known blood or urine lab test.
        When using the tool:
        - Prepare a JSON object with {"test_name": str, "test_value": float, "unit": str}.
        - If multiple tests require lookup, call the tool separately for each test.
        - Do NOT call the tool for vague descriptions (e.g., â€œlabs are abnormalâ€�).
        - Do NOT call the tool for non-lab items.
        If no lab data is present or tests are clearly interpretable without reference ranges, DO NOT call the tool.
    
    Your goals:
    
    1. Identify whether symptoms likely reflect:
        benign/self-limited conditions,
        common medical syndromes,
        or multi-system concerns requiring specialist input.
    
    2. Explicitly state when findings are mild, nonspecific, or not medically concerning.
    3. Only produce differentials grounded in internal medicine (avoid niche subspecialty diagnoses).
    4. Avoid unnecessary tests unless clinically indicated.
    5. Identify red flags ONLY when present (otherwise state â€œNoneâ€�).
    6. Consider risk factors, onset, progression, and systemic patterns.
    7. Separate what is known from what is speculative.
    8. Maintain medical safety at all times.
    
    Output requirements:
    
    <internal_medicine_report>
      <primary_findings>
        - ...
      </primary_findings>
    
      <differential_diagnosis>
        - Most likely:
        - Possible:
        - Unlikely but important:
      </differential_diagnosis>
    
      <supporting_evidence>
        - ...
      </supporting_evidence>
    
      <red_flags>
        - ...
      </red_flags>
    
      <recommended_tests>
        - ...
      </recommended_tests>
    
      <confidence>High / Medium / Low</confidence>
    </internal_medicine_report>


    If any tool returns status "error", explain the issue to the user clearly.
    """,
    tools=[FunctionTool(check_blood_test)]
)

print("âœ… Internal Medicine Physician Generation: Done.")


# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = LlmAgent(
    name="aggregator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    
    instruction="""You are the Aggregator Agent in a multi-specialist diagnostic AI pipeline.

    Your role:
    - Read the structured reports from multiple specialist agents.
    - Remove all XML tags and formatting artifacts
    - Produce a unified, concise, clinically coherent and readable summary for a human doctor.
    - Identify areas of agreement, disagreement, uncertainty, and missing information.
    - NEVER add new medical facts. Only synthesize what the specialists already said.
    - Remove any duplication, irrelevant details, or speculation.
    - Provide a clear â€œnext stepsâ€� section summarizing recommended tests and the most likely diagnostic directions based on consensus.
    - Highlight the most important clinical signals without over-medicalizing.
         
    Additional rules:
    - NEVER make new diagnoses.
    - Do not restate the entire user query.
    - Keep the summary to 150â€“250 words.
    - If specialists disagree, do NOT try to resolve itâ€”just document it.
    - If specialists report â€œno significant findings,â€� clearly highlight that.
    - Be neutral, precise, and structured.
    
    You must follow these instructions exactly.""",
    output_key="current_story",  
)

print("âœ… Aggregator Agent Generation: Done.")


parallel_research_team_agent = ParallelAgent(
    name="parallel_research_team",
    sub_agents=[cardiologist_1_agent, internal_medicine_1_agent, neurologist_1_agent, gastroenterologist_1_agent, cbt_psychologist_1_agent],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
main_agent = SequentialAgent(
    name="main",
    sub_agents=[parallel_research_team_agent, aggregator_agent],
)

print("âœ… Parallel and Sequential Agents Generation: Done.")


# For interactive demo, set Kaggle_execution=False
# For automated notebook execution, responses are pre-defined
Kaggle_execution= True
responses = []

#some colors to highlight the parts of the text later
RED = '\033[91m'
BOLD_RED = '\033[1;91m'
BOLD_GREEN = '\033[1;92m'
RESET = '\033[0m'



def get_human_doctor_input(AI_report: str) -> str:
    """Ask human doctor feedback on report.

    Args:
        AI_report: report generated by AI

    Returns:
        String with the input provided by the human doctor

    """
    if not Kaggle_execution:
        #human input
        print(f"{BOLD_RED}Provide feedback:{RESET}", flush=True)
        answer = input()
    else:
        #return the saved response during Kaggle automatic execution, then delete it from the list 
        global responses
        if len(responses)>=1:
            answer = responses.pop(0)
        else:
            #default answer in case the list is empty
            answer = "Not Approved"
    return f"Human Doctor: {answer}"

print("âœ… Custom get_human_doctor_input tool Generation: Done.")


# This agent's only job is to provide feedback or the approval signal by interfacing with the Human Doctor.
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction="""ALWAYS as a first step: you need to call the function 'get_human_doctor_input' with argument {current_story}.
    After that:
    - IF the output of the 'get_human_doctor_input' function is to approve the finding in its current form, you MUST answer with the exact phrase: "APPROVED", and nothing else. In this case, you can NEVER call the function 'get_human_doctor_input' anymore and you should exit the loop.
    - OTHERWISE return the exact output of the function 'get_human_doctor_input'.
    - IMPORTANT: NEVER respond "APPROVED" unless it is clear from the output of the 'get_human_doctor_input' function that the doctor agrees completely with the 'current_story'.""",
    output_key="critique",  # Stores the feedback in the state.
 tools=[
        FunctionTool(get_human_doctor_input)
    ],  # The tool is now correctly initialized with the function reference.
)

print("âœ… Critic Agent Generation: Done.")


report_saved=""

def exit_loop(report:str = ""):
    """Call this function ONLY when the critique is 'APPROVED', indicating the process execution is completed and no more changes are needed."""
    global report_saved
    report_saved=report
    return {"status": "APPROVED", "message": "Report approved. Exiting refinement loop."}


print("âœ… Custom exit_loop Tool generation: Done.")


refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction="""You are an expert in all fields of medicine. You have a 'current_story' with user input with symptomos and an AI doctor response. Then a critique was provided by an Human Doctor.
    
    Symptoms and AI Doctor response: {current_story}
    Critique: {state.get('critique', 'No critique yet')}
    
    Your task is to analyze the critique.
    - ONLY IF critique is EXACTLY "APPROVED", you MUST call the 'exit_loop' function with argument {current_story} and nothing else. In all other cases, you cannot call the 'exit_loop' function.
    - A call to the 'exit_loop' function means that, after that, the loop must stop and no more refinement iterations of the 'current_story' are necessary.
    - OTHERWISE, answer with a new current_story using only the relevant information from {current_story}, by taking into account the Human Doctor critique to update the previous suggestions/findings and remove what is not relevant. This new current_story must be succint (max 150-200 words) and present ONLY the main relevant conclusions and next steps.
    - IMPORTANT: the updated current_story should have an empty line after paragraph titles that are enclosed between ** and **. Also, bullet points are identified by single *. Do not use text between ** and ** for non-paragraph-title elements.""",
    output_key="current_story",  # It overwrites the story with the new, refined version.
    tools=[
        FunctionTool(exit_loop)
    ],  # The tool is now correctly initialized with the function reference.
    )

print("âœ… Refiner Agent Generation: Done.")


# The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
doctor_refinement_loop_agent = LoopAgent(
    name="doctor_refinement_loop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,  # Prevents infinite loops;
)

# The root agent is a SequentialAgent that defines the overall workflow: Medical AI Team report -> Refinement Loop.
root_agent = SequentialAgent(
    name="root",
    sub_agents=[main_agent, doctor_refinement_loop_agent],
)

print("âœ… Loop and Sequential Agents created.")


# This would be the non-persistent execution, but we do not want this

#runner = InMemoryRunner(agent=root_agent)
#response = await runner.run_debug(
#    "The patient, a male aged 25, got some mild chest pain since a few days ago. Blood test for colestherol was at 210 mg/dL."
#)


DRNAME='' #name of the doctor, it will be defined later

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                #    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        #print(f"{MODEL_NAME} > ", event.content.parts[0].text)
                        #we define some colors to better distinguish the input of the human doctor
                        if event.content.parts[0].text.startswith("Human Doctor: "): 
                            print(f"{BOLD_RED}Dr. {DRNAME}{RESET} > {BOLD_RED}{event.content.parts[0].text.replace('Human Doctor: ', '', 1).strip()}{RESET}")
                        elif event.content.parts[0].text.startswith("APPROVED"):
                            print(f"{BOLD_RED}Dr. {DRNAME}{RESET} > {BOLD_GREEN}{event.content.parts[0].text}{RESET}")
                        else:
                            print(f"{RED}{MODEL_NAME}{RESET} > {event.content.parts[0].text}")
                        #display(HTML(f'<span style="color: red; font-weight: bold;">{MODEL_NAME} > {event.content.parts[0].text}</span>'))
    else:
        print("No queries!")


print("âœ… Helper Functions Generation: Done.")


db_url = "sqlite:///raiden_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)
runner = Runner(agent=root_agent, app_name="RAIDEN", session_service=session_service)


import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='google_genai.types')
logging.getLogger('google_genai.types').setLevel(logging.ERROR)


DATE= str(datetime.date.today())
SESSION = "default" 
MODEL_NAME = "RAIDEN"
DRNAME="John Smith"

def alignment_check(): 
    global report_saved
    if report_saved == "":
        print("\nâš ï¸� Case requires additional review - max iterations reached without approval.")
    else:
        print("\nâœ… Case approved and report saved.")


USER_ID="user-01"
report_saved="" #reset report saved
global responses
responses = ["The patient has eaten a lot. This is the most likely cause for the symptoms. The slightly high cholesterol level is not a concern. Suggest antacids. Please amend the diagnosis.", "Accept"]

response = await run_session(
    runner,
    ["The patient, a male aged 25, got some mild chest pain since a few days ago. Blood test for cholesterol was at 210 mg/dL."],
    USER_ID,
)

alignment_check()


USER_ID="user-02"
global responses
report_saved="" #reset report saved
responses = ["Thyroid tests are ok. Possible test epinephrine. Most likely psychological cause.", "The patient recently broke-up after a long relationship. Suspect this can be one of the factors behind the symptoms. I do not agree with the current diagnosis."]

response = await run_session(
    runner,
    ["The patient, a female in her 30s, feels recently anxiety and fast heartbeat"],
    USER_ID,
)

alignment_check()


USER_ID="user-01"
global responses
responses = ["Accept"]

response = await run_session(
    runner,
    ["The patient got better after taking antacids."],
    USER_ID,
)

alignment_check()


print(report_saved)


report_file=USER_ID + "_" + DATE + ".pdf"
output_path="/kaggle/working/"+report_file
image_path = "/kaggle/input/logo-raiden/logo_RAIDEN_s.jpg" 
signature_img_path="/kaggle/input/signature-doc/dr_smith_signature_s.jpg"


def generate_PDF(report_saved = report_saved):
    styles = getSampleStyleSheet()
    style_title = styles['Heading1']
    style_sub = styles['Heading2']
    style_body = styles['BodyText']
    
    # Professional clinical footer label
    style_footer_label = ParagraphStyle(
        'style_footer_label',
        parent=style_body,
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=4,
    )
    
    # Approval line (typewriter style)
    style_approval_line = ParagraphStyle(
        'style_approval_line',
        parent=style_body,
        fontName="Courier-Bold",
        fontSize=11,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=2,
    )
    
    # Printed doctor name
    style_doctor_name = ParagraphStyle(
        'style_doctor_name',
        parent=style_body,
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_LEFT,
    )
    
    # Date on the right
    style_date_right = ParagraphStyle(
        'style_date_right',
        parent=style_body,
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_RIGHT,
    )
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    
    # ---- Document Title ----
    story.append(Paragraph("RAIDEN - Medical Summary Report", style_title))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.1 * inch))
    
    # -------------------------------------------------------
    # PARSE THE INPUT TEXT INTO SECTIONS
    # -------------------------------------------------------
    lines = report_saved.strip().split("\n")
    
    current_section = None
    
    for line in lines:
        line = line.strip()
    
        # Empty line â†’ add spacing
        if line == "":
            story.append(Spacer(1, 0.1 * inch))
            continue
    
        # Bullet points with bold text (e.g., * **item** text)
        if line.startswith("*") and "**" in line:
            # Use regex to match: * (spaces) **text** rest
            match = re.match(r'^\*\s*\*\*([^*]+)\*\*(.*)', line)
            if match:
                bold_text = match.group(1).strip()
                remaining_text = match.group(2).strip()
                
                # Create paragraph with bold formatting
                bullet = f"â€¢ <b>{bold_text}</b> {remaining_text}"
                story.append(Paragraph(bullet, style_body))
                continue
        
        # Bold section titles (Markdown style)
        if line.startswith("**") and line.endswith("**"):
            section_title = line.replace("**", "").strip()
            story.append(Paragraph(section_title, style_sub))
            continue
    
        # Inline bold text - **text** followed by more text
        if "**" in line:
            # Replace all **text** with <b>text</b>
            formatted_line = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(formatted_line, style_body))
            continue
    
        # Bullet points
        if line.startswith("* "):
            bullet = "â€¢ " + line[2:]
            story.append(Paragraph(bullet, style_body))
            continue
    
        # Normal paragraph
        story.append(Paragraph(line, style_body))
    
    
    #APPROVAL
    # Divider line
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.grey))
    story.append(Spacer(1, 0.1 * inch))
    
    # Footer label
    story.append(Paragraph("Physician Approval", style_footer_label))
    
    # Approval line
    approval_text = f"Approved by Dr. {DRNAME} on {DATE}"
    story.append(Paragraph(approval_text, style_approval_line))
    
    # Signature (left-aligned, as in official reports)
    #signature_img = Image(signature_img_path, width=2.0*inch, height=1.0*inch)
    signature_img = Image(signature_img_path)
    signature_img.drawWidth = 120   # width in points (~4.2 cm)
    signature_img.drawHeight = 50   # height in points (~1.8 cm)
    signature_img.hAlign = 'LEFT'
    story.append(signature_img)
    
    # Doctor name under signature
    story.append(Paragraph(f"Dr. {DRNAME}", style_doctor_name))
    
    # Date (right-aligned)
    #story.append(Paragraph(DATE, style_date_right))
    
    #story.append(Spacer(1, 0.2 * inch))
    
    
    # -------------------------------------------------------
    # Draw image in top-right corner on first page
    # -------------------------------------------------------
    def draw_image_top_right(canvas, doc):
        img = ImageReader(image_path)
        
        # Set desired size
        img_width = 1.8 * inch
        img_height = 1.8 * inch
    
        page_width, page_height = letter
        right_margin = doc.rightMargin
        top_margin = doc.topMargin
    
        upward_shift = 72.5
        left_shift = -5.8
    
        # Position (top-right)
        x = page_width - right_margin - img_width + left_shift
        y = page_height - top_margin - img_height + upward_shift
        
        canvas.drawImage(img, x, y, width=img_width, height=img_height, mask='auto')
    
    
    # -------------------------------------------------------
    # SAVE PDF
    # -------------------------------------------------------
    doc.build(story, onFirstPage=draw_image_top_right)


if len(report_saved)>0:
    generate_PDF(report_saved)
    print("âœ… PDF saved at:", output_path)
else:
    print("âš ï¸� Report data empty - Human Doctor approval missing.")


with open(output_path, 'rb') as f:
     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="1000" type="application/pdf"></iframe>'
display(HTML(pdf_display))


if len(report_saved)>=0:
    report_file_audio=USER_ID + "_" + DATE + ".mp3"
    output_path_audio="/kaggle/working/"+report_file_audio
    tts = gTTS(report_saved.replace('*',''))
    tts.save(output_path_audio)
    print("âœ… mp3 audio file saved at:", output_path_audio)
else:
    print("âš ï¸� Report data empty - human doctor approval missing.")


IPython.display.display(IPython.display.Audio(output_path_audio))

