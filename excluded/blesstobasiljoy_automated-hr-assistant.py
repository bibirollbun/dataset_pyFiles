!pip install -q -U google-generativeai pypdf reportlab


import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import os
import logging
from pypdf import PdfReader
from reportlab.pdfgen import canvas

# 1. SETUP LOGGING (Feature: Observability)
# This tracks what the agents are doing.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [AGENT_LOG] - %(message)s')

# 2. SETUP API KEY
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("System configured successfully.")


def create_dummy_resume(filename="candidate_resume.pdf"):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, "NAME: John Doe")
    c.drawString(100, 780, "EMAIL: john.doe@example.com")
    c.drawString(100, 760, "SKILLS: Python, Data Analysis, Basic HTML")
    c.drawString(100, 740, "EXPERIENCE: 2 years as Junior Analyst.")
    c.drawString(100, 720, "EDUCATION: B.Sc Computer Science.")
    c.save()
    print(f"Created dummy resume: {filename}")

# Create the file
create_dummy_resume()

# Define the Job Description
JOB_DESCRIPTION = """
We are looking for a Senior Python Developer.
Must have:
- 5+ years of experience.
- Deep knowledge of AI Agents and LLMs.
- Leadership experience.
"""


def tool_read_resume(file_path):
    """
    Reads text from a PDF file.
    Args: file_path (str)
    Returns: Extracted text or error message.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logging.info(f"Tool used: Read PDF ({len(text)} chars extracted)")
        return text
    except Exception as e:
        logging.error(f"Tool Error: {e}")
        return "Error reading file."


def run_hr_agent(resume_path, job_desc):
    logging.info("--- STARTING AGENT WORKFLOW ---")
    
    # 1. USE TOOL
    resume_text = tool_read_resume(resume_path)
    
    # 2. AGENT 1: THE EVALUATOR (Reasoning)
    logging.info("Agent 1 (Evaluator) starting...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    eval_prompt = f"""
    Act as a strict HR Manager.
    
    JOB DESCRIPTION:
    {job_desc}
    
    RESUME TEXT:
    {resume_text}
    
    Task:
    1. Score the candidate from 0 to 100 based on the job match.
    2. Explain your reasoning in 1 sentence.
    
    Output format:
    SCORE: [number]
    REASON: [text]
    """
    
    eval_response = model.generate_content(eval_prompt)
    evaluation_result = eval_response.text
    logging.info(f"Evaluation complete. Result:\n{evaluation_result}")
    
    # 3. AGENT 2: THE COMMUNICATOR (Action)
    logging.info("Agent 2 (Communicator) starting...")
    
    email_prompt = f"""
    You are an HR Assistant. Read this evaluation:
    {evaluation_result}
    
    Instruction:
    - If the SCORE is < 50, write a polite rejection email.
    - If the SCORE is >= 50, write an enthusiastic interview invitation.
    
    Output ONLY the email body.
    """
    
    email_response = model.generate_content(email_prompt)
    
    logging.info("--- WORKFLOW FINISHED ---")
    return email_response.text


def run_hr_agent(resume_path, job_desc):
    logging.info("--- STARTING AGENT WORKFLOW ---")
    
    # 1. USE TOOL
    resume_text = tool_read_resume(resume_path)
    
    # 2. AGENT 1: THE EVALUATOR (Reasoning)
    logging.info("Agent 1 (Evaluator) starting...")
    
    # UPDATED: Using the specific model from your list
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    eval_prompt = f"""
    Act as a strict HR Manager.
    
    JOB DESCRIPTION:
    {job_desc}
    
    RESUME TEXT:
    {resume_text}
    
    Task:
    1. Score the candidate from 0 to 100 based on the job match.
    2. Explain your reasoning in 1 sentence.
    
    Output format:
    SCORE: [number]
    REASON: [text]
    """
    
    eval_response = model.generate_content(eval_prompt)
    evaluation_result = eval_response.text
    logging.info(f"Evaluation complete. Result:\n{evaluation_result}")
    
    # 3. AGENT 2: THE COMMUNICATOR (Action)
    logging.info("Agent 2 (Communicator) starting...")
    
    email_prompt = f"""
    You are an HR Assistant. Read this evaluation:
    {evaluation_result}
    
    Instruction:
    - If the SCORE is < 50, write a polite rejection email.
    - If the SCORE is >= 50, write an enthusiastic interview invitation.
    
    Output ONLY the email body.
    """
    
    email_response = model.generate_content(email_prompt)
    
    logging.info("--- WORKFLOW FINISHED ---")
    return email_response.text


# Run the system
final_email = run_hr_agent("candidate_resume.pdf", JOB_DESCRIPTION)

print("\n" + "="*30)
print("FINAL AGENT OUTPUT (EMAIL DRAFT)")
print("="*30)
print(final_email)

