import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API key ready.")
except Exception as e:
    print(f"Error: {e}")


from google.genai import types
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, FunctionTool

print("ADK dependencies imported.")


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

print("Retry configurations defined for reliability.")


model = Gemini(model_name="gemini-1.5-flash", retry_options=retry_config)
agent = Agent(
    name="trend_researcher",
    model=model, 
    description="An agent for researching career trends",
    instruction="Research top skills for the query using available tools. Provide direct answers with key details only.",
    tools=[google_search]
)
runner = InMemoryRunner(agent=agent)
response = await runner.run_debug("AI engineer skills 2025")
print(response)


# Researcher Agent
agent_researcher = Agent(
    name="researcher",
    model=model,
    description="Researches trends.",
    instruction="Provide concise top 5 trends summary. Use search tool.",
    tools=[google_search]
)

# Gap Analyzer Agent
agent_analyzer = Agent(
    name="analyzer",
    model=model,
    description="Analyzes skills gaps.",
    instruction="Provide concise gap list from resume and trends.",
    tools=[]
)

# Planner Agent
agent_planner = Agent(
    name="planner",
    model=model,
    description="Generates plans.",
    instruction="Provide concise 6-month plan with 5 mielestone based on goal and gaps.",
    tools=[]
)

print("Agents defined.")


# Custom Resume Parser Tool -> Extracts text from PDF for gap analysis

!pip install PyPDF2

import PyPDF2

def parse_resume(path: str) -> str:
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ' '.join(page.extract_text() for page in reader.pages)
        return text[:1000] # Cap for shorter responses

agent_analyzer.tools = [parse_resume]
print("Parser tool added to Analyzer.")


# Test Researcher - Checks trend research
runner = InMemoryRunner(agent=agent_researcher)
response = await runner.run_debug("AI skills 2025")
print(response)


# Test Analyzer - Checks gap analysis (use mock or file path)
runner = InMemoryRunner(agent=agent_analyzer)
mock_resume = "Skills: Python, ML basics" # Or parse_resume('path/to/mock_resume.pdf')
mock_trends = "AI agents, RAG"
response = await runner.run_debug(f"Resume: {mock_resume}\nTrends: {mock_trends}")
print(response)


# Test Planner - Checks plan generation
runner = InMemoryRunner(agent=agent_planner)
mock_goal = "AI Engineer"
mock_gaps = "Need RAG, deployment"
response = await runner.run_debug(f"Goal: {mock_goal}\n{mock_gaps}")
print(response)


from google.adk.runners import InMemoryRunner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.in_memory_session_service import AlreadyExistsError

# Initialize shared session service for persistent conversations across runs
session_service = InMemorySessionService()
print("InMemorySessionService initialized for persistent sessions.")


# Update planner instruction to handle follow-ups and progress updates
agent_planner.instruction = (
    "Provide concise 6-month plan with 5 milestones based on goal and gaps. "
    "If the user provides progress updates, adjust the plan accordingly, "
    "removing completed milestones and refining remaining ones."
)
print("Planner instruction updated for follow-up support.")


!pip install reportlab -q
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
from google.genai import types

async def career_flow(goal: str, resume_path: str, session_id: str = "career_session", user_id: str = "default_user"):
    # Researcher: Get trends (no session needed for one-shot research)
    runner_researcher = InMemoryRunner(agent=agent_researcher)
    trends_response = await runner_researcher.run_debug(goal + " skills and trends 2025")
    # Extract the final response text (assuming the last event is the model response)
    trends = trends_response[-1].content.parts[0].text if trends_response else "No trends found."

    # Parse resume
    resume_text = parse_resume(resume_path)

    # Analyzer: Identify gaps (no session needed for one-shot analysis)
    runner_analyzer = InMemoryRunner(agent=agent_analyzer)
    analyze_input = f"Resume: {resume_text}\nTrends: {trends}"
    gaps_response = await runner_analyzer.run_debug(analyze_input)
    gaps = gaps_response[-1].content.parts[0].text if gaps_response else "No gaps identified."

    # Planner: Generate or update plan (with session for persistence)
    runner_planner = Runner(app_name="career_planner_app", agent=agent_planner, session_service=session_service)
    
    # Create session if it doesn't exist (idempotent)
    try:
        await session_service.create_session(
            app_name="career_planner_app",
            user_id=user_id,
            session_id=session_id  # Optional, but specify to use custom ID
        )
    except AlreadyExistsError:
        pass  # Session already exists, proceed
    
    plan_content = types.Content(
        role="user",
        parts=[types.Part(text=f"Goal: {goal}\n{gaps}")]
    )
    plan_response = []
    async for event in runner_planner.run_async(user_id=user_id, session_id=session_id, new_message=plan_content):
        plan_response.append(event)
    plan = plan_response[-1].content.parts[0].text if plan_response else "No plan generated."

    return plan

# Test orchestration with valid mock PDF (using reportlab to create a simple valid PDF)
mock_resume_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
c = canvas.Canvas(mock_resume_path, pagesize=letter)
c.drawString(100, 750, "Skills: Python, ML basics. Experience: Data analysis projects.")
c.save()

test_plan = await career_flow("Become an AI Engineer", mock_resume_path)
print("Test Plan:\n", test_plan)


from google.genai import types
from google.adk.sessions.in_memory_session_service import AlreadyExistsError

# Demonstrate follow-up with the same session (progress tracking)
user_id = "test_user"
session_id = "test_session_1"

# Initial plan generation
runner_planner = Runner(app_name="career_planner_app", agent=agent_planner, session_service=session_service)

# Create session if it doesn't exist (idempotent)
try:
    await session_service.create_session(
        app_name="career_planner_app",
        user_id=user_id,
        session_id=session_id  # Optional, but specify to use custom ID
    )
except AlreadyExistsError:
    pass  # Session already exists, proceed

initial_content = types.Content(
    role="user",
    parts=[types.Part(text="Goal: AI Engineer\nNeed RAG, deployment, advanced ML")]
)
initial_response = []
async for event in runner_planner.run_async(user_id=user_id, session_id=session_id, new_message=initial_content):
    initial_response.append(event)
initial_plan = initial_response[-1].content.parts[0].text
print("Initial Plan:\n", initial_plan)

# Follow-up: Simulate user returning with progress
followup_content = types.Content(
    role="user",
    parts=[types.Part(text="I completed Milestone 1 (foundations). Update the plan.")]
)
followup_response = []
async for event in runner_planner.run_async(user_id=user_id, session_id=session_id, new_message=followup_content):
    followup_response.append(event)
updated_plan = followup_response[-1].content.parts[0].text
print("\nUpdated Plan after Progress:\n", updated_plan)


# Define Judge Agent for evaluation
agent_judge = Agent(
    name="judge",
    model=model,
    description="Evaluates career plans.",
    instruction=(
        "Score the provided plan on usefulness (how actionable it is) and relevance "
        "(how well it addresses the goal and gaps) from 1-10. Provide a brief reason for each score."
    ),
    tools=[]
)

# Function to evaluate a plan
async def evaluate_plan(plan: str, goal: str, gaps: str):
    runner_judge = InMemoryRunner(agent=agent_judge)
    eval_input = f"Plan: {plan}\nGoal: {goal}\nGaps: {gaps}"
    eval_response = await runner_judge.run_debug(eval_input)
    score = eval_response[-1].content.parts[0].text if eval_response else "Evaluation failed."
    return score

# Test evaluation with mock data from previous outputs
mock_goal = "AI Engineer"
mock_gaps = "Need RAG, deployment, advanced ML"
mock_plan = """Here is a concise 6-month plan with 5 milestones to achieve your goal of becoming an AI Engineer, focusing on RAG, deployment, and advanced ML:

---

**AI Engineer 6-Month Plan**

**Goal:** Become a proficient AI Engineer with skills in RAG, deployment, and advanced ML.

**Milestones:**

1.  **Foundational RAG & Basic Deployment (Months 1-1.5):**
    *   **Focus:** Understand RAG architecture. Implement a basic RAG system (LangChain/LlamaIndex) with a local LLM or API.
    *   **Deployment:** Deploy a simple ML model (e.g., sentiment classifier) locally using Flask/FastAPI and containerize with Docker.
2.  **Advanced RAG & Cloud Deployment (Months 1.5-3):**
    *   **Focus:** Explore advanced RAG techniques: vector databases (Pinecone/Chroma/Weaviate), query optimization, and RAG evaluation.
    *   **Deployment:** Deploy your RAG system and ML models to a cloud platform (AWS/GCP/Azure) using serverless functions or managed services. Implement basic CI/CD for deployments.
3.  **Advanced Machine Learning & Optimization (Months 3-4.5):**
    *   **Focus:** Deep dive into Transformer architectures (BERT, GPT variants). Experiment with fine-tuning pre-trained models for specific tasks.
    *   **Optimization:** Learn model optimization techniques (quantization, pruning) and performance evaluation metrics.
4.  **MLOps & Production-Ready AI Systems (Months 4.5-5.5):**
    *   **Focus:** Implement MLOps practices: model versioning (MLflow/DVC), experiment tracking.
    *   **Production:** Set up monitoring, logging, and alerting for deployed AI services. Understand scaling strategies and ensure robust deployments.
5.  **End-to-End AI Project & Portfolio (Months 5.5-6):**
    *   **Focus:** Develop a comprehensive, end-to-end AI project incorporating advanced RAG, fine-tuned ML models, and cloud deployment with MLOps principles.
    *   **Portfolio:** Document projects, prepare for technical interviews, and refine your resume to showcase your AI engineering capabilities.

---

Please provide updates on your progress, and I will adjust the plan accordingly!"""
test_score = await evaluate_plan(mock_plan, mock_goal, mock_gaps)
print("Test Evaluation:\n", test_score)


!pip install gradio -q
import gradio as gr

# Gradio function wrapper (handles file upload)
async def generate_plan(goal: str, resume_file):
    if resume_file is None:
        return "Please upload a resume PDF."
    # Save uploaded file temporarily
    resume_path = resume_file.name
    plan = await career_flow(goal, resume_path)
    # Evaluate the plan (use mock gaps for simplicity; in full impl, extract from analyzer)
    gaps = "Need RAG, deployment, advanced ML"  # Replace with actual gaps extraction if needed
    score = await evaluate_plan(plan, goal, gaps)
    return f"**Generated Plan:**\n{plan}\n\n**Evaluation Score:**\n{score}"

# Create Gradio interface
demo = gr.Interface(
    fn=generate_plan,
    inputs=[
        gr.Textbox(label="Career Goal", placeholder="e.g., Become an AI Engineer"),
        gr.File(label="Upload Resume PDF", file_types=[".pdf"])
    ],
    outputs=gr.Markdown(label="Personalized Career Plan & Evaluation"),
    title="Career Co-Pilot AI",
    description="Enter your career goal and upload your resume to get a personalized 6-month plan."
)

demo.launch(share=True)


# Test edge cases
# Edge case 1: Empty goal
try:
    await career_flow("", "mock_resume.pdf")
except Exception as e:
    print(f"Handled empty goal error: {e}")

# Edge case 2: Invalid resume path
try:
    await career_flow("AI Engineer", "invalid_path.pdf")
except Exception as e:
    print(f"Handled invalid resume error: {e}")

# Demo video                    ---> Manually record a 1-min screen capture of Gradio UI and upload to YouTube.
#                               ---> Add link here: e.g., https://youtu.be/EXAMPLE_LINK (replace with real link)

print("Notebook polished. Ready for submission!")

