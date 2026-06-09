pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Auth Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# âœ… Install and configure Gemini API
!pip install -q google-generativeai
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name='models/gemini-2.0-flash')


pip install pdfplumber


import pdfplumber

def pdf_to_string(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

job_description = pdf_to_string("/kaggle/input/job-description/Job_D.pdf")
resume = pdf_to_string("/kaggle/input/uday-resume/Uday_Resume.pdf")


print(job_description[:500])


print(resume[:500])


# âœ… Step 1: Score match and suggest bullet points
prompt = f'''
Compare the resume and job description below.

1. List matching skills
2. Provide a match score (0â€“10)
3. Suggest 2 bullet points to add to the resume

Job Description:
{job_description}

Resume:
{resume}
'''

response = model.generate_content(prompt)
print(response.text)


# âœ… Step 2: Generate structured JSON output
json_prompt = f'''
Generate a JSON object with:
- job_title
- company
- match_score
- resume_bullets
- custom_cover_letter

Use the job description and resume.

Job:
{job_description}
Resume:
{resume}
'''

response = model.generate_content(json_prompt)
print(response.text)

