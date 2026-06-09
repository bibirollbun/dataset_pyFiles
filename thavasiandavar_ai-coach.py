# --- CELL 1: Install Dependencies ---
!pip install google-genai

# --- CELL 2: Setup & Imports ---
import os
from google import genai
from google.genai import types

# [IMPORTANT] REPLACE WITH YOUR FRIEND'S API KEY
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
print("Setup complete.")

# --- CELL 3: Define Custom Tool ---

# This tool simulates a database that knows job salaries
def get_job_salary_data(role: str):
    """Retrieves average salary and growth data for a job title."""
    # Mock database
    market_data = {
        "Python Developer": "Avg Salary: $120k/year. Demand: High",
        "Data Analyst": "Avg Salary: $95k/year. Demand: Very High",
        "Project Manager": "Avg Salary: $110k/year. Demand: Moderate",
        "Web Designer": "Avg Salary: $85k/year. Demand: Stable"
    }
    # Default fallback
    return market_data.get(role, "Avg Salary: $90k/year. Demand: Stable")

# Manual Tool Definition (Safe method)
salary_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_job_salary_data",
            description="Get salary and demand info for a specific job role.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "role": types.Schema(
                        type="STRING",
                        description="The job title (e.g. Python Developer)"
                    )
                },
                required=["role"]
            )
        )
    ]
)

print("Tool 'get_job_salary_data' ready.")

# --- CELL 4: Agent Class (Robust Version) ---
class CareerAgent:
    def __init__(self, name, instruction, tools=None):
        self.name = name
        self.instruction = instruction
        self.tools = tools
        # Using the standard model
        self.model = "gemini-1.5-flash" 

    def run(self, prompt):
        full_content = f"SYSTEM ROLE: {self.instruction}\nUSER INPUT: {prompt}"
        
        config = None
        if self.tools:
            config = types.GenerateContentConfig(tools=self.tools)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[full_content],
                config=config
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"

# --- CELL 5: Run the Project ---
def run_career_demo():
    print("--- Starting AI Career Coach ---")
    
    # Agent 1: The Resume Scanner
    # Extracts skills from raw text
    scanner = CareerAgent(
        name="ResumeScanner",
        instruction="You are a hiring expert. Read the user's resume text and output ONLY a list of their top 3 technical skills."
    )
    
    # Agent 2: The Job Matcher
    # Takes skills, picks a job title, and checks salary using the tool
    matcher = CareerAgent(
        name="JobMatcher",
        instruction="You are a career coach. Based on the provided skills, suggest ONE specific job title. Then ALWAYS use the tool 'get_job_salary_data' to check the salary for that title.",
        tools=[salary_tool]
    )

    # 1. Simulate User Resume
    user_resume = "I have been coding for 3 years using Python and Pandas. I love analyzing data and creating charts."
    print(f"\nUser Resume Snippet: '{user_resume}'")

    # 2. Agent 1 Runs
    print(f"\n[{scanner.name} is analyzing...]")
    scan_result = scanner.run(user_resume)
    skills_found = scan_result.text
    print(f"Skills Extracted: {skills_found}")

    # 3. Agent 2 Runs (Sequential)
    print(f"\n[{matcher.name} is finding a job match...]")
    match_response = matcher.run(f"Suggest a job for these skills: {skills_found}")

    # 4. Handle Tool Use Manually (To ensure it works)
    final_output = match_response.text or ""
    
    if hasattr(match_response, 'function_calls') and match_response.function_calls:
        for fc in match_response.function_calls:
            if fc.name == "get_job_salary_data":
                job_arg = fc.args["role"]
                print(f" > Tool Triggered: Checking market data for '{job_arg}'...")
                
                # Run the Python function
                data = get_job_salary_data(job_arg)
                print(f" > Tool Result: {data}")
                
                final_output = f"Recommended Role: {job_arg}\nMarket Data: {data}\nReasoning: Matches your Python/Data skills."

    print(f"\n--- FINAL CAREER ADVICE ---\n{final_output}")

# Execute
run_career_demo()


