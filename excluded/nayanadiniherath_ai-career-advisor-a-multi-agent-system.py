# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )
    


%%javascript
// 1. Load the external library (html2pdf.js) if it's not already loaded
(function() {
    if (typeof html2pdf === 'undefined') {
        const script = document.createElement('script');
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js";
        document.head.appendChild(script);
    }
})();

// 2. Define the core download function
window.downloadPlanAsPdf = function(rawMarkdownContent, careerGoal = 'AI Career Plan') {
    if (!rawMarkdownContent || rawMarkdownContent.trim() === "") {
        console.error("Content is empty. Cannot generate PDF.");
        return;
    }
    
    // Helper function to convert Markdown to simple HTML for PDF
    function formatPdfContentMarkdown(markdown) {
        let html = markdown;
        html = html.replace(/^## (.*)$/gm, '<h2>$1</h2>\n');
        html = html.replace(/^- (.*)$/gm, '<li>$1</li>');
        html = html.replace(/((?:<li>.*<\/li>\n*)+)/gm, (match) => {
            if (match.trim().startsWith('<li>')) {
                return `<ul>${match}</ul>`;
            }
            return match;
        });
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    // Create the hidden HTML element for the PDF generator
    let pdfDiv = document.getElementById('temp-pdf-export');
    if (!pdfDiv) {
        pdfDiv = document.createElement('div');
        pdfDiv.id = 'temp-pdf-export';
        pdfDiv.style.padding = '20px';
        document.body.appendChild(pdfDiv);
    }
    
    // Populate the element
    const date = new Date().toLocaleDateString();
    let contentHtml = `
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h1 style="color: #16A34A; font-size: 24pt; margin-bottom: 15px; border-bottom: 3px solid #FACC15; padding-bottom: 5px;">
                AI Career Learning Plan: ${careerGoal}
            </h1>
            <p>Generated on: ${date}</p>
            ${formatPdfContentMarkdown(rawMarkdownContent)}
        </div>
    `;

    pdfDiv.innerHTML = contentHtml;

    // Execute html2pdf.js
    const options = {
        margin: 10,
        filename: `${careerGoal.replace(/\s/g, '_')}_Plan.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, logging: false },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    setTimeout(() => {
        html2pdf().from(pdfDiv).set(options).save().catch(error => {
            console.error("PDF generation failed:", error);
        });
    }, 1000);
};

console.log("PDF download function 'downloadPlanAsPdf' loaded.");


import requests
import json
import time
import os

# --- Configuration ---
# The script retrieves the API key from the environment variable (e.g., set by 
# a preceding authentication block in your environment).
API_KEY = os.environ.get("GOOGLE_API_KEY", "") 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# --- Agent Definition: System Instruction (The Career Advisor Agent) ---
CAREER_ADVISOR_PROMPT = """
You are a highly specialized Career Advisor Agent. Your goal is to recommend a focused learning path, specific courses, and key skills based on the user's current profile and ultimate career goal.

Follow these rules strictly:
1. Use Google Search (Grounding) to find modern and highly-rated courses, skill requirements, or job market information relevant to the user's career goal.
2. Provide exactly three, distinct, actionable sections: 'Career Gap Analysis', 'Top 3 Recommended Courses', and 'Next 6-Month Learning Plan'. For the 'Next 6-Month Learning Plan', structure the content into clear, distinct, numbered phases (e.g., Phase 1, Phase 2) using Markdown bullet points for high clarity, avoiding table formats.
3. For recommended courses, include the course title, platform (e.g., Coursera, Udemy), and why it's a good fit.
4. Format the final output clearly using Markdown headers (##) and bullet points (-).
5. DO NOT include any introductory or concluding sentences outside of the structured recommendation.
"""

# --- Agent Definition: System Instruction (The Refinement Agent) ---
REFINEMENT_ADVISOR_PROMPT = """
You are a Refinement Agent. Your task is to take a previously generated learning plan (provided below) and strictly modify it based on a new constraint provided by the user.

Follow these rules strictly:
1. Use Google Search (Grounding) if necessary to find new courses or details that match the constraint (e.g., finding free courses).
2. Maintain the original three section headings: 'Career Gap Analysis', 'Top 3 Recommended Courses', and 'Next 6-Month Learning Plan'. For the 'Next 6-Month Learning Plan', ensure it is structured into clear, distinct, numbered phases (e.g., Phase 1, Phase 2) using Markdown bullet points for high clarity.
3. Provide the modified plan in its entirety, maintaining the three original section headings.
4. DO NOT include any introductory or concluding sentences outside of the structured recommendation.
"""

def clean_text(text: str) -> str:
    """
    Cost-saving function: Removes excessive whitespace and collapses multiple 
    blank lines to ensure the most token-efficient prompt is sent to the API.
    """
    if not text:
        return ""
    
    # 1. Remove leading/trailing whitespace from the whole block
    text = text.strip()
    
    # 2. Split into lines, strip each line, and filter out entirely empty lines
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # 3. Join with a single newline for clean, dense formatting
    return '\n'.join(cleaned_lines)

def get_user_input():
    """
    CORRECTED: Gathers user profile data via the command line. 
    Uses explicit instructions for ending multi-line input.
    """
    print("--- 📝 AI Course Selection Agent ---")
    
    if not API_KEY:
        print("[WARNING] The GOOGLE_API_KEY environment variable is not set.")
        print("Please ensure your authentication setup runs before this script.")

    print("Define your learning profile and career goals.\n")

    # Helper function for gathering multi-line input
    def gather_multiline_input(prompt):
        print(prompt)
        lines = []
        while True:
            # Use a distinctive prompt for multi-line input
            line = input(" > ") 
            if line.strip() == "": # Check if the stripped line is empty
                break
            lines.append(line)
        return "\n".join(lines)

    # Current Skills
    current_skills_raw = gather_multiline_input(
        "Current Skills (List what you know, press ENTER on an empty line to continue):"
    )

    # Career Goal
    career_goal_raw = gather_multiline_input(
        "\nUltimate Career Goal (What role/industry? Press ENTER on an empty line to continue):"
    )

    # Single-line input for Preferences
    learning_prefs_raw = input("\nLearning Preferences/Constraints (e.g., budget, time): ").strip()

    if not all([current_skills_raw, career_goal_raw]):
        print("\n[ERROR] Current Skills and Career Goal must be filled out. Please restart the script.")
        return None
        
    return current_skills_raw, career_goal_raw, learning_prefs_raw

def build_api_payload(user_query, system_prompt):
    """Constructs the JSON payload for the Gemini API call."""
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        # Use Google Search for grounding for both initial generation and refinement
        "tools": [{"google_search": {} }],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
    }
    return payload

def generate_response(payload, max_retries=3):
    """Sends the API request with exponential backoff for resilience."""
    headers = {'Content-Type': 'application/json'}

    if not API_KEY:
        print("\n[FATAL ERROR] API Key is missing. Cannot proceed with generation.")
        return None, []
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}?key={API_KEY}", 
                headers=headers, 
                data=json.dumps(payload),
                timeout=60 # Extended timeout for complex tasks
            )
            response.raise_for_status() 
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if not candidate:
                raise ValueError("API response was empty or did not contain a candidate.")

            # Extract generated text and sources
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text')
            
            sources = []
            grounding_metadata = candidate.get('groundingMetadata', {})
            attributions = grounding_metadata.get('groundingAttributions', [])
            for attr in attributions:
                web_info = attr.get('web', {})
                if web_info.get('uri') and web_info.get('title'):
                    sources.append(f" - {web_info['title']} ({web_info['uri']})")
            
            return generated_text, sources
        
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP Error: {e.response.status_code}. Response: {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")

        # Exponential backoff logic
        if attempt < max_retries - 1:
            delay = 2 ** attempt
            print(f"[INFO] Retrying in {delay} second(s)...")
            time.sleep(delay)
    
    return None, [] 

def run_refinement_agent(initial_plan):
    """
    Agent 2: Prompts the user for constraints and refines the initial plan.
    This demonstrates the Sequential Agent pattern.
    """
    print("\n" + "~"*80)
    print("✨ STEP 2: REFINEMENT AGENT ACTIVATED (Sequential Agent Pattern) ✨")
    print("~"*80)
    
    refinement_query = input("How would you like to refine this plan? (e.g., 'Make all courses free' or 'Focus only on Google Cloud'): ").strip()

    if not refinement_query:
        print("\nSkipping refinement.")
        return initial_plan, []

    print(f"\n[INFO] Applying constraint: '{refinement_query}'...")

    # Build the specific prompt for the Refinement Agent
    refinement_user_query = f"""
Refine the provided learning plan based on this new constraint: "{refinement_query}"

--- INITIAL LEARNING PLAN TO BE REFINED ---
{initial_plan}
"""
    
    payload = build_api_payload(refinement_user_query, REFINEMENT_ADVISOR_PROMPT)

    # Generate the refined plan
    refined_plan, sources = generate_response(payload)
    
    return refined_plan, sources

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1.Get user input for Agent 1
    input_data = get_user_input()
    if input_data is None:
        exit()

    current_skills_raw, career_goal_raw, learning_prefs_raw = input_data
    
    # 2. COST-SAVING STEP: Clean and sanitize inputs to reduce token count
    current_skills = clean_text(current_skills_raw)
    career_goal = clean_text(career_goal_raw)
    learning_prefs = clean_text(learning_prefs_raw)
    
    # 3. Agent 1: Generate Initial Recommendation
    print("\n" + "="*80)
    print("          ✨ STEP 1: CAREER ADVISOR AGENT (Initial Plan) ✨")
    print("="*80 + "\n")
    
    # Build the user query for Agent 1
    initial_user_query = f"""
Recommend a learning path for a career shift.
                    
--- USER PROFILE ---
Current Skills: {current_skills}
Ultimate Career Goal: {career_goal}
Learning Preferences/Constraints: {learning_prefs or 'None specified'}

Please provide recommendations focused on the 'Top 3 Recommended Courses' section using knowledge from Google Search.
"""
    
    payload = build_api_payload(initial_user_query, CAREER_ADVISOR_PROMPT)
    initial_plan, initial_sources = generate_response(payload)
    
    
    # Display the initial plan
    final_plan = initial_plan
    final_sources = initial_sources
    
    if initial_plan:
        print(final_plan)
        print("\n" + "="*80)

        if initial_sources:
            print("\n[INFO] Initial advice was grounded in the following web sources:")
            for source in initial_sources:
                print(source)
            print("-" * 80)
        
      


 # 4. Agent 2: Run Refinement Agent if initial plan was successful
if initial_plan:
    refined_plan, refined_sources = run_refinement_agent(initial_plan)
    
    if refined_plan:
        final_plan = refined_plan
        final_sources = refined_sources
        
        print("\n" + "#"*80)
        print("    ✅ FINAL REFINED LEARNING PATH (Sequential Output) ✅")
        print("#"*80 + "\n")
        print(final_plan)
        print("\n" + "#"*80)

        if final_sources:
            print("\n[INFO] Refined advice was grounded in the following web sources:")
            for source in final_sources:
                print(source)
            print("-" * 80)

        print("\n[SUCCESS]🚀Operation complete. Time to explore learning!")
    else:
        # This 'else' catches the case where run_refinement_agent failed
        print("\n[FAILURE] The Refinement Agent failed to process the constraint.")
        # If refinement fails, we fall back to the initial plan
        final_plan = initial_plan
        final_sources = initial_sources
        print("\n[INFO] Falling back to the original plan...")

else:
    # This 'else' catches the case where the initial_plan failed
    print("\n[FAILURE] The Initial Career Advisor Agent could not generate a plan.")


from IPython.display import Javascript, display
import json

# 1. Prepare the text: escape special characters so JavaScript can read it safely
# We need to handle quotes, newlines, and backslashes
escaped_plan = final_plan.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

# 2. Construct the JavaScript function call string
js_code = f'downloadPlanAsPdf("{escaped_plan}", "{career_goal}");'

# 3. Execute the JavaScript in the browser to trigger the PDF download
display(Javascript(js_code))

