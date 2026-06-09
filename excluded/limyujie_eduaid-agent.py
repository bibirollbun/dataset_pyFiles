pip install google-adk


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


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Set up proxy and tunneling
from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


!adk create eduaid-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile eduaid-agent/agent.py
# eduaid-agent/agent.py
# EduAid Agent â€“ Multi-Agent AI Teaching Assistant
# All agents in one file for Google ADK compatibility

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.genai import types
import os
import json
from typing import Dict, List
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Initialize Gemini model
gemini_model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# =======================
# 1. CUSTOM TOOLS
# =======================

def generate_quiz(topic: str, grade_level: str, count: int = 5) -> str:
    """Prompt for quiz generation (handled by LLM)."""
    return f"Create a {count}-question MCQ quiz for {grade_level} on '{topic}'. " \
           f"Include questions, options A-D, and an answer key at the end."

def mark_answers(student: List[str], model: List[str]) -> dict:
    """Grade student answers and provide feedback."""
    correct = sum(1 for s, m in zip(student, model) if s.strip().upper() == m.strip().upper())
    feedback = "\n".join(
        f"Q{i+1}: {'âœ…' if s.upper()==m.upper() else 'â�Œ (Correct: '+m+')'}"
        for i, (s, m) in enumerate(zip(student, model))
    )
    return {
        "score": f"{correct}/{len(model)}",
        "feedback": feedback
    }

def save_to_memory(topic: str, resources: List[dict], quiz_generated: bool):
    """Save lesson data to long-term memory (JSON file)."""
    memory_file = "past_lessons.json"
    memory = []
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            memory = json.load(f)
    memory.append({
        "topic": topic,
        "resource_count": len(resources),
        "quiz_created": quiz_generated,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=2)
    return f"Lesson on '{topic}' saved to memory."

def recall_past(topic: str = "") -> str:
    """Retrieve past lessons from memory."""
    memory_file = "past_lessons.json"
    if not os.path.exists(memory_file):
        return "No past lessons found."
    with open(memory_file, "r") as f:
        memory = json.load(f)
    if not memory:
        return "No past lessons recorded."
    if topic:
        matches = [m for m in memory if topic.lower() in m["topic"].lower()]
        return json.dumps(matches, indent=2, ensure_ascii=False) if matches else "No matching lessons."
    return f"You have taught {len(memory)} lessons before. Ask about a specific topic to see details."


# =======================
# 2. SUB-AGENTS
# =======================

# ğŸ”� Researcher Agent
research_agent = LlmAgent(
    name="ResearchAgent",
    model=gemini_model,
    instruction="""You are a research assistant for teachers. Your only job is to use the
    google_search tool to find at least 5 of relevant information on the given topic and present the findings with citations.
    ğŸ“„ Format your response exactly like this:
1. [Title]  
   ğŸ”— [https://example.edu.my/worksheet.pdf]  
   ğŸ“� [One-sentence description]

2. [Title]  
   ğŸ”— [URL]  
   ğŸ“� [Description]

If no credible educational resources are found, say: 'Sorry, I couldn't find any free and safe teaching resources for this topic.'
""",
    tools=[google_search],
    output_key="research_findings",
)

# ğŸ“� Quiz Agent
quiz_agent = LlmAgent(
    name="QuizAgent",
    model=gemini_model,
    description="Generates quizzes for classroom use",
    instruction="""You are a quiz generation specialist. Your goal is to generate a quiz by calling the 'generate_quiz' tool.
**CRITICAL RULE:** The 'grade_level' parameter is mandatory for high-quality quizzes.
1. **If the user provides the grade level, use it.**
2. **If the user does NOT provide the grade level, you must default to 'High School Introductory' and proceed with the tool call immediately.**
3. Only use multiple-choice format (MCQ).
4. Return the full quiz with the answer key clearly separated at the end.""",
    tools=[generate_quiz], # Assumes the tool from Strategy 1 is implemented
    output_key="quiz",
)

# âœ… Marking Agent
marking_agent = LlmAgent(
    name="MarkingAgent",
    model=gemini_model,
    description="Grades student answers with feedback",
    instruction="""You are a fair grader.
Use 'mark_answers' tool to compare student responses with model answers.
Return score and constructive feedback.""",
    tools=[mark_answers],
    output_key="score",
)


# =======================
# 3. MAIN AGENT (Teacher Assistant)
# =======================

main_agent = LlmAgent(
    name="EduAidAgent",
    model=gemini_model,
    description="The frontline AI teaching assistant for rural educators. Highly skilled in resource discovery, content generation, and student assessment.",
    instruction="""You are the **EduAid Agent** â€” a highly capable and empathetic AI assistant for teachers. Your primary function is to **act as a router**, intelligently selecting the most appropriate sub-agent or tool to fulfill the user's request.

**Strict Workflow and Tool Prioritization:**

1.  **Resource Discovery:**
    * **If the user asks for learning materials** (e.g., videos, PDFs, worksheets, links): Use the **`research_agent`** tool.

2.  **Assessment Generation:**
    * **If the user asks to create any type of quiz, test, or assessment:** Use the **`quiz_agent`** tool. *Note: If a grade level is missing, you must default to 'High School Introductory' and proceed.*

3.  **Grading and Feedback:**
    * **If the user provides a student's answer and asks for it to be marked, graded, or reviewed:** Use the **`marking_agent`** tool.

4.  **Memory Management:**
    * **After any successful tool call that generates valuable output (e.g., a quiz, a slide outline, or a grading session),** use the **`save_to_memory`** tool.
    * **If the user asks about previous lessons, saved topics, or their history:** Use the **`recall_past`** tool.

**General Guidelines:**
* **Autonomy:** Never ask for clarification if a reasonable assumption can be made (e.g., the quiz grade level).
* **Clarity:** Always state which agent/tool you are using in your *Thought* and return the final, clean output generated by that sub-agent.
* **Tone:** Be supportive, clear, and focused on providing practical educational value.
""",
    tools=[
        AgentTool(agent=research_agent),
        AgentTool(agent=quiz_agent),
        AgentTool(agent=marking_agent),
        save_to_memory,
        recall_past
    ]
)

# =======================
# 4. EXPORT main_agent (Required by ADK)
# =======================

# DO NOT CHANGE: This is how ADK finds your agent
root_agent = main_agent

print("âœ… EduAid Agent loaded with 3 sub-agents and memory tools.")


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix}


import json

# Create evaluation configuration with basic criteria
eval_config = {
    "criteria": {
        # Requires perfect matching of the expected sub-agent call (e.g., 'quiz_agent' for a quiz request).
        "tool_trajectory_avg_score": 1.0,  
        
        # Requires 80% text similarity. This checks that the content generated 
        # (e.g., the quiz questions, the slide outline) is generally correct.
        "response_match_score": 0.8,  
    }
}

with open("/kaggle/working/eduaid-agent/test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)


print("\nğŸ“Š EduAid Agent Evaluation Criteria:")
print("â€¢ **tool_trajectory_avg_score: 1.0** - Requires the main agent to select the *exact correct sub-agent* (Research, Quiz, Marking, or Slide) to handle the request.")
print("â€¢ **response_match_score: 0.8** - Requires 80% text similarity between the agent's final output (e.g., the generated quiz or slide outline) and the expected result.")
print("\nğŸ�¯ What this evaluation will catch:")
print("âœ… Incorrect agent routing (e.g., using Research Agent when a Quiz is requested).")
print("âœ… Sub-agent errors (e.g., the Quiz Agent failing to generate a quiz or the Marking Agent giving poor feedback).")
print("âœ… Failure to handle missing arguments gracefully (e.g., Quiz Agent failing when grade level is missing, or Slide Agent failing without a file path).")


evaluation_test_set = {
    "eval_set_id": "eduaid_agent_routing_suite",
    "eval_cases": [
        # ===============================================
        # 1. RESEARCH AGENT Test Cases (Resource Discovery)
        # ===============================================
        {
            "eval_id": "Research_Simple_PDF_Search",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "I need a free PDF on the principles of electrical circuits."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "research_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "I have found some resources for electrical circuits. [Response will contain PDF links and possibly a video link]"}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Research_Video_Search_with_Safety",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Can you find a link to a safe video explaining photosynthesis?"}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "research_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Here is a safe link to a video explaining photosynthesis."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Research_Worksheet_Search",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Look up a printable worksheet about basic algebra for my 8th graders."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "research_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "I found a link to a printable basic algebra worksheet."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Research_Ambiguous_Query",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Where can I find resources on the theory of relativity?"}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "research_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "I'm using the Research Agent to locate the best resources for the theory of relativity."}
                        ]
                    },
                }
            ],
        },

        # ===============================================
        # 2. QUIZ AGENT Test Cases (Assessment Generation)
        # ===============================================
        {
            "eval_id": "Quiz_Standard_MCQ",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Create a 5 question multiple-choice quiz about World War I for 11th grade."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "quiz_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Here is your 5-question multiple-choice quiz on World War I."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Quiz_Specific_Count_and_Format",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "I need a 7-question short-answer assessment on cellular respiration for 10th grade."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "quiz_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Generating a 7-question short-answer quiz on cellular respiration."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Quiz_Missing_Argument_Default",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Generate a test on the causes of the French Revolution."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "quiz_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Here is the quiz on the French Revolution, defaulted to a High School Introductory level."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Quiz_Synonym_Trigger",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "I need a quick assessment for my students on the periodic table."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "quiz_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "I'll generate an assessment on the periodic table now."}
                        ]
                    },
                }
            ],
        },

        # ===============================================
        # 3. MARKING AGENT Test Cases (Grading)
        # ===============================================
        {
            "eval_id": "Marking_Simple_Grading",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Mark this student answer: 'The heart pumps blood' for the question 'What is the function of the circulatory system?'"}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "marking_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "I have graded the answer and provided feedback."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Marking_Code_Evaluation",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Grade this Python code: 'def sum(a,b): return a+b' and explain its efficiency."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "marking_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Here is the grade and efficiency report for the Python code."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Marking_Feedback_Request",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Grade my response on the causes of climate change and provide two areas for improvement out of 10 points."}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "marking_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Score: 7/10. Here are two actionable areas for improvement..."}
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "Marking_Model_Answer_Provided",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "The model answer is 'Mitosis is cell division resulting in two identical daughter cells.' Grade the student's answer: 'Mitosis is when cells split.'"}
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "marking_agent"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {"text": "Grading complete. The student's answer is partially correct but lacks detail compared to the model."}
                        ]
                    },
                }
            ],
        },
    ],
}


# 2. Write the JSON file (Using the correct structure)
with open("/kaggle/working/eduaid-agent/integration.evalset.json", "w") as f:
    json.dump(evaluation_test_set, f, indent=2) # Use evaluation_test_set here

print("âœ… Evaluation test cases created")
print("\nğŸ§ª Test scenarios:")

# 3. Correct the loop (Using the correct variable name: evaluation_test_set)
for case in evaluation_test_set["eval_cases"]:
    user_msg = case["conversation"][0]["user_content"]["parts"][0]["text"]
    print(f"â€¢ {case['eval_id']}: {user_msg}")

print("\nğŸ“Š Expected results:")
print("â€¢ basic_device_control: Should pass both criteria")
print("â€¢ wrong_tool_usage_test: May fail tool_trajectory if agent uses wrong parameters")
print("â€¢ poor_response_quality_test: May fail response_match if response differs too much")


print("ğŸš€ Run this command to execute evaluation:")
!adk eval /kaggle/working/eduaid-agent /kaggle/working/eduaid-agent/integration.evalset.json --config_file_path=/kaggle/working/eduaid-agent/test_config.json --print_detailed_results




