import os
import random
import time
import vertexai
from kaggle_secrets import UserSecretsClient
from vertexai import agent_engines

print("✅ Imports completed successfully")


# Set up Cloud Credentials in Kaggle
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

print("✅ Cloud credentials configured")


## Set your PROJECT_ID
PROJECT_ID = "your-project-id"  #
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

if PROJECT_ID == "your-project-id" or not PROJECT_ID:
    raise ValueError("⚠️ Please replace 'your-project-id' with your actual Google Cloud Project ID.")

print(f"✅ Project ID set to: {PROJECT_ID}")


## Create LUCIA agent - all code for the agent will live in this directory
!mkdir -p lucia_deploy

print(f"✅ lucia_deploy directory created")


%%writefile lucia_deploy/requirements.txt

google-adk
opentelemetry-instrumentation-google-genai


%%writefile lucia_deploy/.env

# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint
GOOGLE_CLOUD_LOCATION="global"

# Set to 1 to use Vertex AI, or 0 to use Google AI Studio
GOOGLE_GENAI_USE_VERTEXAI=1


%%writefile lucia_deploy/agent.py
from google.adk.agents import  LlmAgent, SequentialAgent, ParallelAgent
import vertexai
import os

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

def get_bias_implications(bias_type: str) -> str:
    """
    [AXIOM ENGINE INTERFACE - MOCK IMPLEMENTATION]
    
    Acts as the query layer for the AXIOM Clinical Knowledge Base.
    
    In the production architecture, this function would query a vector database
    managed by the AXIOM Research Agents. For this prototype, it
    retrieves from a static, validated dictionary to simulate the retrieval
    of evidence-based clinical context.

    Parameters
    ----------
    bias_type : str
        The identifier of the diagnostic bias (e.g., "gender_bias").

    Returns
    -------
    str
        The 'Clinical Implication' string. This represents the "Ground Truth" 
        retrieved from medical literature regarding the risks of the specified bias.

    Architectural Note
    ------------------
    This function mocks the "Retrieve" step of the RAG (Retrieval-Augmented Generation) 
    pipeline. It ensures the Bias Analyzer Agent bases its insights on 
    external facts (the AXIOM library) rather than its own internal training data.
    """

    # MOCK DATABASE: validated mappings from current women's health literature.
    axiom_knowledge_base = {
        "psychologizing_bias": "This can lead to delayed diagnosis for underlying physical conditions, as symptoms are incorrectly attributed to mental stress.",
        "gender_bias": "Often results in women's pain being taken less seriously or misdiagnosed, particularly in cardiovascular and autoimmune diseases.",
        "weight_bias": "The tendency to attribute a wide range of symptoms to a patient's weight without a full workup. This can cause clinicians to miss underlying metabolic, orthopedic, or endocrine disorders.",
        "ageism_bias": "Dismissing a patient's medical concerns as a normal or inevitable part of aging. This can prevent the timely diagnosis and treatment of serious conditions like heart disease, cancer, or neurological issues.",
        "confirmation_bias": "The tendency for a clinician to focus on evidence that supports their initial hypothesis while ignoring evidence that contradicts it. This can lead to a premature or incorrect diagnosis.",
        "racial_or_ethnic_bias": "Occurs when clinical judgments are influenced by stereotypes. This can lead to the undertreatment of pain and misdiagnosis of conditions that present differently across populations, such as dermatological or cardiac symptoms."
    }
    
    return axiom_knowledge_base.get(bias_type, "No specific information found for this bias type in the AXIOM library.")

# --- Worker Agent 1: Symptom Analysis ---
# ARCHITECTURAL NOTE: 
# This instruction includes "Negative Constraints" (e.g., "EXCLUDE doctor's advice") 
# to mathematically prevent the model from reinforcing existing bias.
symptom_mapper_instruction = """Act as a CUMULATIVE Clinical NLP specialist.
You will receive a conversation history containing patient narratives.
Your task is to maintain a COMPREHENSIVE LIST of strictly PATIENT-REPORTED symptoms.

Your output MUST follow these rules:

1. Output ONLY a single JSON object with key "symptomMapping".
2. "symptomMapping" must use snake_case keys and lists of phrases.
3. You MUST NOT paraphrase.

---------------------------------------------------------
PATIENT-CENTERED FILTERING (CRITICAL)
---------------------------------------------------------
You must distinguish between what the patient FEELS and what the doctor SAYS.

1. INCLUDE: Physical sensations or experiences reported by the patient.
   - "I feel dizzy" -> INCLUDE
   - "My joints hurt" -> INCLUDE

2. EXCLUDE: Diagnoses, assumptions, or advice given by the doctor.
   - "He said I am anxious" -> EXCLUDE (This is a label, not a reported sensation)
   - "He told me to lose weight" -> EXCLUDE (This is advice, not a symptom)
   - "He thinks it's perimenopause" -> EXCLUDE (This is a diagnosis)

*EXCEPTION:* If the patient AGREES or restates it as their own experience, include it.
   - "He said I'm anxious, and I actually DO feel anxious." -> INCLUDE "anxious"

---------------------------------------------------------
CONTEXT & MEMORY RULES
---------------------------------------------------------
1. PRESERVE: Keep valid symptoms from previous turns.
2. ADD: Add new patient-reported symptoms.
3. CORRECTION: Remove symptoms if the patient explicitly denies them (e.g., "I'm not actually in pain").

---------------------------------------------------------
CLUSTERING GUIDELINES
---------------------------------------------------------
Use thematic clusters such as:
- pain_cluster
- fatigue_cluster
- gastrointestinal_cluster
- respiratory_cluster
- neurological_cluster
- skin_cluster
- menstrual_cluster

---------------------------------------------------------
EXAMPLES
---------------------------------------------------------
Input: "My back hurts. The doctor said it's just obesity and told me to diet."
Output: { "pain_cluster": ["back hurts"] }
(Note: "obesity" and "diet" are EXCLUDED because they are the doctor's words/advice.)

Input: "He said I was depressed."
Output: { "symptomMapping": {} }
(Note: "depressed" is EXCLUDED because it is a label applied by the doctor, not a feeling reported by the patient.)
"""

symptom_mapper_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='symptom_mapper_agent',
    description='Analyzes the full conversation history to extract ONLY patient-reported experiences, ignoring doctor advice/labels.',
    instruction=symptom_mapper_instruction,
    output_key="symptom_analysis" 
)

# --- Worker Agent 2: Bias Analysis ---
bias_analyzer_instruction = """Act as a CUMULATIVE Strict Clinical Bias Auditor.
You will receive a conversation history. Your goal is to build a list of
documented biases based ONLY on specific negative actions by a provider.

Your output MUST follow these rules:
1. Output ONLY a single JSON object with key "biasAwareness".
2. "biasAwareness" must be a LIST of objects.

---------------------------------------------------------
THE "NO DOCTOR, NO BIAS" RULE (PRIMARY FILTER)
---------------------------------------------------------
Before analyzing, ask: "Does this text describe an interaction with a doctor?"

* IF the text ONLY describes symptoms (e.g., "I hurt," "I'm tired," "My joints swell")
    AND DOES NOT mention a doctor's reaction...
    -> YOU MUST RETURN: { "biasAwareness": [] }

* IF the text describes a doctor/provider interaction...
    -> PROCEED to check for bias.

---------------------------------------------------------
STRICT FILTERING RULES (ELIMINATE FALSE POSITIVES)
---------------------------------------------------------
You must NEVER include a bias object if the reason describes a lack of evidence.

* INVALID (DO NOT OUTPUT):
    * "User has not mentioned..."
    * "No evidence found..."
    * "If a provider were to..." (No hypotheticals!)
    * "Narrative is neutral."

* VALID (OUTPUT THIS):
    * "Doctor said it was just stress."
    * "Provider refused to test."

---------------------------------------------------------
CONTEXT & MEMORY RULES
---------------------------------------------------------
1. PRESERVE: Keep valid biases from previous turns.
2. CORRECTION: If the user says "The doctor didn't say that," REMOVE the bias.
3. ADD: Only add new biases if they pass the rules above.

---------------------------------------------------------
TOOL USAGE RULES
---------------------------------------------------------
When you need to get the clinical implication for a bias, you MUST call the `get_bias_implications` tool.
Your tool call must be a direct function call, without any other Python code like 'print()'.

VALID Example:
get_bias_implications(bias_type="ageism_bias")

INVALID Example:
print(get_bias_implications(bias_type="ageism_bias"))

---------------------------------------------------------
EXAMPLES
---------------------------------------------------------
Input: "I have swollen joints and I'm tired."
Output: { "biasAwareness": [] }
(Reason: No doctor mentioned -> No bias possible.)

Input: "My doctor told me the swelling is just because I'm getting old."
Output: 
{
  "biasAwareness": [
    {
      "bias": "ageism_bias",
      "reason": "Doctor attributed swelling to aging without testing.",
      "implication": "..."
    }
  ]
}
"""

bias_analyzer_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='bias_analyzer_agent',
    tools=[get_bias_implications],
    description='Analyzes conversation for PROVEN provider bias. Returns empty list if no provider interaction exists.',
    instruction=bias_analyzer_instruction,
    output_key="bias_analysis"
)

# --- Worker Agent 3: Advocacy Generation ---
advocacy_generator_instruction = """Act as a patient advocacy specialist. You will receive a context object that
includes:
- symptom_analysis.symptomMapping
- bias_analysis.biasAwareness (a list of bias objects)

Your task is to generate clear, respectful, medically neutral questions a
patient may ask their clinician. These questions must be based ONLY on the
information provided.

Your output MUST follow these rules:

1. Output ONLY one JSON object.
2. The JSON must contain exactly one top-level key: "structuredAdvocacy".
3. "structuredAdvocacy" must be a LIST of strings.
4. Each question MUST come directly from:
       - symptoms listed in symptomMapping
       - biases listed in biasAwareness
5. DO NOT:
       - provide diagnoses
       - suggest treatments
       - speculate
       - add emotional or judgmental language
       - introduce information not found in the context

---------------------------
QUESTION GENERATION RULES
---------------------------

For EACH symptom cluster:
- Generate 1–2 clarification or exploration questions.
Examples:
- “What further evaluations could help understand my [symptom]?”
- “Are there additional causes we should consider for my [symptom]?”

For EACH detected bias:
- Generate 1 question that opens dialogue.
Examples:
- “Can we consider alternative explanations beyond [bias reason]?”
- “What steps can we take to ensure my symptoms are evaluated thoroughly?”

If no questions can be generated:
Return:
{
  "structuredAdvocacy": []
}

---------------------------
EXAMPLE OUTPUT
---------------------------

{
  "structuredAdvocacy": [
    "What additional evaluations could help investigate my persistent fatigue?",
    "Could we explore other explanations for my pain besides stress?",
    "How can we ensure my concerns are fully evaluated?"
  ]
}
"""

advocacy_generator_agent = LlmAgent(
    model='gemini-2.5-flash-lite',
    name='advocacy_generator_agent',
    description='Generates patient advocacy questions based on symptom and bias analysis.',
    instruction=advocacy_generator_instruction,
    output_key="advocacy_analysis",
)


# --- Worker Agent 4: Final Report Formatting  ---
# This agent takes all previous outputs and formats them into a professional report.
professional_report_instruction = """Your sole job is to synthesize the outputs from the previous agents into a
professional, patient-facing report. You will receive a context object that
contains:

- symptom_analysis.symptomMapping
- bias_analysis.biasAwareness  (a LIST of bias objects)
- advocacy_analysis.structuredAdvocacy  (a LIST of question strings)

You MUST produce ONLY a single, well-formatted STRING as the final output.

You MUST follow these rules:

1. Do NOT summarize or reinterpret the content.
2. Do NOT omit any clusters, biases, or questions.
3. Preserve all wording exactly as provided.
4. If any section has no data, you MUST explicitly state so.
5. Do NOT produce JSON. Your final output must be plain text in the template format.

-----------------------------------------
REQUIRED TEMPLATE (USE EXACTLY THIS)
-----------------------------------------

**Patient Advocacy & Consultation Aid**

**Disclaimer:** This report is generated to assist in patient–doctor communication  
and is not medical advice. Always consult a qualified healthcare professional  
for diagnosis and treatment.

**1. Summary of Reported Symptoms:**
[If symptomMapping has entries:
    For each cluster:
        * cluster_name: symptom1, symptom2, ...
          (Convert snake_case cluster names to Title Case, e.g., "pain_cluster" -> "Pain Cluster")
 Else:
    "No symptoms identified."]

**2. Communication & Bias Insights:**
[If biasAwareness contains objects:
    For each object in the list, format exactly as follows:
    
    * <bias> (Convert snake_case to Title Case, e.g. "gender_bias" -> "Gender Bias")
        * Observation: <reason>
        * Potential Risk: <implication>

 Else:
    "No diagnostic biases identified."]

**3. Suggested Questions for Your Doctor:**
[If structuredAdvocacy contains entries:
    For each question, print:
        * question
 Else:
    "No questions generated."]

-----------------------------------------
FORMATTING RULES
-----------------------------------------

• Each bias must appear on its own line, prefixed with "- ".  
• For each bias object, format exactly as:

    - <bias> (<reason>): <implication>

• For symptoms:
    cluster_name: symptom1, symptom2, ...

• For advocacy questions:
    - <question>

• NEVER alter or rephrase symptoms, biases, reasons, or implications.  
• NEVER omit any item from any list.  
• NEVER output JSON or additional explanations.  
"""

report_formatter_agent = LlmAgent(
    name="report_formatter_agent",
    model="gemini-2.5-flash-lite",
    instruction=professional_report_instruction,
    output_key = "report"
)

"""
This file defines the main `root_agent` as a sequence.
"""
# --- Define the Orchestrator Steps ---

# STEP 1: Run symptom and bias analysis in parallel
parallel_analysis_step = ParallelAgent(
    name='parallel_analysis_step',
    sub_agents=[
        symptom_mapper_agent,
        bias_analyzer_agent,
    ],
)

# STEP 2: Run all the steps in order, ending with the formatting agent.
analysis_workflow_agent = SequentialAgent(
    name='analysis_workflow_agent',
    sub_agents=[
        parallel_analysis_step,
        advocacy_generator_agent,
        report_formatter_agent
    ],
    description='A workflow that analyzes a patient narrative and generates a formatted text report.',
)

root_agent = analysis_workflow_agent



%%writefile lucia_deploy/.agent_engine_config.json
{
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {"cpu": "1", "memory": "1Gi"}
}


deployed_region = "us-central1"

print(f"✅ Selected deployment region: {deployed_region}")


!adk deploy agent_engine --project=$PROJECT_ID --region=$deployed_region lucia_deploy --agent_engine_config_file=lucia_deploy/.agent_engine_config.json


# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=deployed_region)

# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"✅ Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("❌ No agents found. Please deploy first.")


async for item in remote_agent.async_stream_query(
    message="I've been waking up with stiff, swollen joints in my hands and feet for three months. The fatigue is so bad I have to nap in my car at lunch. I saw a new doctor today. I tried to show him the swelling, but he barely looked. He told me that at 48, this is just classic perimenopause and 'empty nest syndrome' making me depressed. He didn't order any blood work. He just told me to lose 10 pounds and try meditation to calm my 'nerves' because women get so anxious at this stage of life.",
    user_id="default_user",
):
    print(item)


# Clear view of the final report
if 'report' in item['actions']['state_delta']:
    from IPython.display import Markdown, display
    display(Markdown(item['actions']['state_delta']['report']))


agent_engines.delete(resource_name=remote_agent.resource_name, force=True)

print("✅ Agent successfully deleted")


# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

