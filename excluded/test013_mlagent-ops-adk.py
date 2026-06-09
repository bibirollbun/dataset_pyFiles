
import os
import json
import textwrap
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown

# ADK Imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

from google.adk.models.google_llm import Gemini 
from google.genai import types



#1. SETUP & CONFIGURATION
 
# Define ANSI colors for visual clearance in the console
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Setup API Key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print(f"{Colors.GREEN}âœ… Gemini API key setup complete.{Colors.RESET}")
except Exception as e:
    print(f"{Colors.WARNING}ğŸ”‘ Authentication Warning: Ensure 'GOOGLE_API_KEY' is in Secrets. Details: {e}{Colors.RESET}")

print(f"{Colors.GREEN}âœ… ADK components imported successfully.{Colors.RESET}")



#2. AGENT DEFINITIONS (MLOPS PIPELINE)
 
# --- Agent 1: Data Ingestion ---
data_crawler_agent = Agent(
    name="DataCrawlerAgent",
    model="gemini-2.5-flash-lite",#String is fine here (uses default settings)
    instruction="""You are a Data Pipeline Engineer. 
    1. Identify the dataset or problem domain specified in the user's prompt (e.g., "CIFAR-10", "IMDB Sentiment", "California Housing").
    2. Use google_search to find key characteristics: dataset size, number of features/classes, and data type (image, text, tabular).
    3. Format your output as a structured "Data Ingestion Report".""",
    tools=[google_search],
    output_key="data_report",
)

# --- Agent 2: Model Testing ---
# Schema includes generic metrics compatible with classification and regression
MODEL_METRICS_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "model_name": types.Schema(type=types.Type.STRING, description="Name of the model tested (e.g., BERT, ResNet, XGBoost)."),
        "task_type": types.Schema(type=types.Type.STRING, description="Task type: Classification, Regression, etc."),
        "primary_metric_name": types.Schema(type=types.Type.STRING, description="Name of the main metric (Accuracy, RMSE, IoU)."),
        "primary_metric_value": types.Schema(type=types.Type.NUMBER, description="Value of the main metric."),
        "training_time_hours": types.Schema(type=types.Type.NUMBER, description="Simulated training time."),
        "deployment_readiness_score": types.Schema(type=types.Type.NUMBER, description="Score 0-10 on readiness."),
    },
    required=["model_name", "task_type", "primary_metric_name", "primary_metric_value", "training_time_hours"]
)

structured_output_model = Gemini(
    model="gemini-2.5-flash-lite",
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MODEL_METRICS_SCHEMA,
    )
)

model_tester_agent = Agent(
    name="ModelTesterAgent",
    model=structured_output_model,#Pass the configured model object, not a string
    instruction="""You are a Model Training Engineer. 
    Based on the data report: {data_report}, select a standard model architecture suitable for this specific data type (e.g., CNN for images, Transformer for text, GBDT for tabular).
    Simulate a training run and output realistic performance metrics in the required JSON format. 
    Be realistic: do not give perfect scores.""",
    output_key="test_results_json",
)

# --- Agent 3: MLOps Reporting ---
mlops_reporter_agent = Agent(
    name="MLOpsReporterAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are an MLOps Analyst. Review the metrics: {test_results_json}.
    Write a "Go/No-Go" deployment decision report.
    Crucially, suggest the specific visualization plots needed for this model type in the TensorFlow Viewer/Dashboard (e.g., 'AUC Curve' for classification vs 'Residual Plot' for regression).""",
    output_key="mlops_report",
)

# --- Final Pipeline Definition ---
mlops_pipeline_agent = SequentialAgent(
    name="UniversalMLOpsPipeline",
    sub_agents=[data_crawler_agent, model_tester_agent, mlops_reporter_agent],
)
print(f"{Colors.GREEN}âœ… UniversalMLOpsPipeline created successfully.{Colors.RESET}")



#3. ENHANCED EXECUTION LOGIC
 
async def run_and_visualize_pipeline(runner, prompt, use_case_name):
    print(f"\n{Colors.HEADER}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}ğŸš€ STARTING USE CASE: {use_case_name}{Colors.RESET}")
    print(f"{Colors.HEADER}{'='*80}{Colors.RESET}")
    print(f"ğŸ“� Prompt: {prompt}\n")

    try:
     #Execute the pipeline
        response_events = await runner.run_debug(prompt)
    except Exception as e:
        print(f"{Colors.WARNING}â�Œ Pipeline Runtime Error: {e}{Colors.RESET}")
        return

    events = response_events if isinstance(response_events, list) else [response_events]
    
    print(f"{Colors.BOLD}--- ğŸ“œ Step-by-Step Execution Trace ---{Colors.RESET}")
    
    step_count = 1
    final_output_text = ""

    for event in events:
        agent_name = getattr(event, 'author', 'System')
        
     #Extract text content
        content_text = ""
        if hasattr(event, 'content') and event.content and event.content.parts:
            text_parts = [p.text for p in event.content.parts if hasattr(p, 'text') and p.text]
            content_text = "\n".join(text_parts)

     #--- VISUALIZATION LOGIC ---
        
     #1. Handle Tool Calls (Cyan)
        if hasattr(event, 'content') and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    print(f"\n{Colors.CYAN}ğŸ› ï¸�  [Tool Call] Agent '{agent_name}' is using tool: {part.function_call.name}{Colors.RESET}")

     #2. Handle Agent Output (Green/Blue)
        if content_text:
         #Check if output is JSON (for ModelTesterAgent) and format it
            try:
                json_obj = json.loads(content_text)
                formatted_json = json.dumps(json_obj, indent=2)
                print(f"\n{Colors.BLUE}â�¡ï¸�  [Step {step_count}] Agent: {agent_name} (JSON Output){Colors.RESET}")
                print(f"{Colors.GREEN}{formatted_json}{Colors.RESET}")
            except json.JSONDecodeError:
             #Standard text output
                print(f"\n{Colors.BLUE}â�¡ï¸�  [Step {step_count}] Agent: {agent_name}{Colors.RESET}")
                print(f"{content_text.strip()}")
            
            final_output_text = content_text
            step_count += 1

 #--- FINAL REPORT ---
    print(f"\n{Colors.HEADER}âœ… PIPELINE COMPLETE: {use_case_name}{Colors.RESET}\n")
    if final_output_text:
        display(Markdown(f"### ğŸ“Š Final Executive Report: {use_case_name}\n\n{final_output_text}"))



#4. EXECUTION LOOPS (3 DISTINCT USE CASES)
runner = InMemoryRunner(agent=mlops_pipeline_agent)



# Use Case 1: Vision (Computer Vision)
await run_and_visualize_pipeline(
    runner, 
    prompt="Run MLOps pipeline for the CIFAR-10 image dataset.", 
    use_case_name="Computer Vision (CIFAR-10)"
)



# Use Case 2: NLP (Sentiment Analysis)
await run_and_visualize_pipeline(
    runner, 
    prompt="Run MLOps pipeline for the IMDB Movie Review dataset for sentiment analysis.", 
    use_case_name="NLP (IMDB Sentiment)"
)



# Use Case 3: Tabular (Predictive Regression)
await run_and_visualize_pipeline(
    runner, 
    prompt="Run MLOps pipeline for the California Housing dataset to predict median house prices.", 
    use_case_name="Tabular Regression (Housing Prices)"
)

