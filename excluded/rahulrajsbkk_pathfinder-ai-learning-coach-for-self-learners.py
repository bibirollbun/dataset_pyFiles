##### Pathfinder Learning Coach â€“ Capstone Notebook
##### Inspired by the Kaggle 5â€‘Day Agents Intensive materials




import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        "ğŸ”‘ Authentication Error: Please make sure you have added "
        "'GOOGLE_API_KEY' to your Kaggle secrets. Details:", e
    )




from datetime import datetime
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

warnings.filterwarnings("ignore")

print("âœ… Core libraries imported.")




# Base Gemini model used for all agents
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

print(f"âœ… Configuration ready. Using model: {GEMINI_MODEL_NAME}")




# HTML Export Tool V2 (Detailed Course Generation)


def export_course_as_html(
    course_title: str,
    modules: List[Dict[str, Any]],
    output_filename: str = "pathfinder_course.html",
    user_id: str = "demo_user",
) -> str:
    """
    Export a detailed course plan as a styled HTML report.
    
    Args:
        course_title: Title of the course (e.g. "Python for Data Analysis")
        modules: List of module dictionaries, each containing:
                 - title: str
                 - explanation: str
                 - exercises: List[str]
                 - resources: List[str]
                 - quiz: List[Dict] (question, options, answer)
        output_filename: Name of the HTML file to generate
        user_id: Optional user identifier (not used in this notebook, but kept for future use)
    
    Returns:
        A confirmation message with the file path
    """
    print(f"DEBUG: export_course_as_html called for '{course_title}' with {len(modules)} modules.")

    # Generate HTML
    try:
        html_content = _generate_course_html(
            course_title=course_title,
            modules=modules
        )
        print("DEBUG: HTML content generated successfully.")
    except Exception as e:
        print(f"DEBUG: Error generating HTML content: {e}")
        raise e
    
    # Write to file
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"DEBUG: File written to {output_filename}")
    except Exception as e:
        print(f"DEBUG: Error writing to file: {e}")
        raise e
    
    return f"âœ… Course HTML generated: {output_filename}"


def _generate_course_html(
    course_title: str,
    modules: List[Dict[str, Any]]
) -> str:
    """Generate the complete HTML document with embedded CSS/JS."""
    
    modules_html = ""
    for idx, module in enumerate(modules, 1):
        # Format exercises
        exercises_html = ""
        for ex in module.get("exercises", []):
            exercises_html += f"<li>{ex}</li>"
            
        # Format resources
        resources_html = ""
        for res in module.get("resources", []):
            resources_html += f"<li><a href='#' onclick='return false;'>{res}</a></li>"
            
        # Format quiz
        quiz_html = ""
        for q_idx, q in enumerate(module.get("quiz", []), 1):
            options_html = ""
            correct_answer = q.get("answer", "")
            for opt in q.get("options", []):
                is_correct = opt == correct_answer
                options_html += f"""
                <div class="option" onclick="checkAnswer(this, {str(is_correct).lower()})">
                    {opt}
                </div>
                """
            
            quiz_html += f"""
            <div class="quiz-question">
                <p><strong>Q{q_idx}:</strong> {q.get("question")}</p>
                <div class="options">
                    {options_html}
                </div>
                <div class="feedback"></div>
            </div>
            """

        modules_html += f"""
        <div class="module-card">
            <div class="module-header" onclick="toggleModule(this)">
                <h2>Module {idx}: {module.get("title")}</h2>
                <span class="toggle-icon">â–¼</span>
            </div>
            <div class="module-content">
                <div class="section">
                    <h3>ğŸ“– Explanation</h3>
                    <div class="explanation">
                        {module.get("explanation", "").replace(chr(10), "<br>")}
                    </div>
                </div>
                
                <div class="section">
                    <h3>ğŸ’» Practice Exercises</h3>
                    <ul>
                        {exercises_html}
                    </ul>
                </div>
                
                <div class="section">
                    <h3>ğŸ”— Resources</h3>
                    <ul>
                        {resources_html}
                    </ul>
                </div>
                
                <div class="section">
                    <h3>ğŸ§  Knowledge Check</h3>
                    <div class="quiz-container">
                        {quiz_html}
                    </div>
                </div>
            </div>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{course_title} - Pathfinder Course</title>
    <style>
        :root {{
            --primary: #667eea;
            --secondary: #764ba2;
            --bg: #f5f7fa;
            --text: #333;
            --success: #28a745;
            --error: #dc3545;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding-bottom: 4rem;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 0 1rem;
        }}
        
        .module-card {{
            background: white;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            overflow: hidden;
            transition: transform 0.2s;
        }}
        
        .module-header {{
            padding: 1.5rem;
            background: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
        }}
        
        .module-header:hover {{
            background: #f8f9fa;
        }}
        
        .module-header h2 {{
            font-size: 1.2rem;
            color: var(--secondary);
        }}
        
        .module-content {{
            padding: 1.5rem;
            display: none; /* Hidden by default */
        }}
        
        .module-content.active {{
            display: block;
            animation: slideDown 0.3s ease-out;
        }}
        
        @keyframes slideDown {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .section {{
            margin-bottom: 2rem;
        }}
        
        .section h3 {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            margin-bottom: 1rem;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
            display: inline-block;
        }}
        
        .explanation {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--primary);
        }}
        
        ul {{
            list-style-position: inside;
            padding-left: 1rem;
        }}
        
        li {{
            margin-bottom: 0.5rem;
        }}
        
        /* Quiz Styles */
        .quiz-question {{
            background: #fff;
            border: 1px solid #eee;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        
        .option {{
            padding: 0.8rem;
            margin: 0.5rem 0;
            border: 1px solid #ddd;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .option:hover {{
            background: #f0f0f0;
        }}
        
        .option.correct {{
            background: #d4edda;
            border-color: var(--success);
            color: #155724;
        }}
        
        .option.incorrect {{
            background: #f8d7da;
            border-color: var(--error);
            color: #721c24;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
    <script>
        function toggleModule(header) {{
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            
            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                icon.textContent = 'â–¼';
            }} else {{
                content.classList.add('active');
                icon.textContent = 'â–²';
            }}
        }}
        
        function checkAnswer(element, isCorrect) {{
            const parent = element.parentElement;
            // Disable all options in this question
            const options = parent.querySelectorAll('.option');
            options.forEach(opt => {{
                opt.style.pointerEvents = 'none';
                opt.style.opacity = '0.7';
            }});
            
            if (isCorrect) {{
                element.classList.add('correct');
                element.style.opacity = '1';
                element.innerHTML += ' âœ…';
            }} else {{
                element.classList.add('incorrect');
                element.style.opacity = '1';
                element.innerHTML += ' â�Œ';
            }}
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>ğŸ§­ {course_title}</h1>
        <p>Your Personalized Learning Course</p>
    </div>
    
    <div class="container">
        {modules_html}
        
        <div class="footer">
            <p>Generated by Pathfinder Learning Coach â€¢ {datetime.now().strftime("%B %d, %Y")}</p>
        </div>
    </div>
</body>
</html>"""
    
    return html


print("âœ… Advanced HTML export tool defined.")





# In-memory log of study sessions for this notebook run
SESSION_LOG: List[Dict[str, Any]] = []


def record_session_summary(user_id: str, topic: str, summary: str, quiz_score: float) -> str:
    """Record a short summary and score for a finished study session.

    This keeps a lightweight in-memory log for the current run.
    """
    entry = {
        "user_id": "demo_user",
        "topic": topic,
        "summary": summary,
        "quiz_score": quiz_score,
    }
    SESSION_LOG.append(entry)

    return "Session summary recorded."


print("âœ… Tools defined: record_session_summary()")





def make_planner_instruction() -> str:
    return """
    You are a learning planner agent.

    The user will give you:
    - A learning goal (e.g. "learn Python for data analysis")
    - A time horizon (e.g. "4 weeks")
    - Rough time available per week

    Your job is to:
    1. Break the goal into weekly milestones
    2. For each week, list 3-5 concrete study tasks
    3. Map tasks to simple internal topic keys like:
       - python_basics
       - python_loops
       - data_analysis
    4. Output a concise plan that the tutor, quiz, and export agents can follow.

    IMPORTANT: If the user does not specify a time horizon or hours per week,
    assume "4 weeks" and "5 hours per week" by default.
    DO NOT ask for clarification. ALWAYS generate a plan immediately.

    Be realistic and keep the plan achievable.
    """


def make_resource_instruction() -> str:
    return """
    You are a resource curator agent.

    Given a roadmap from the planner, align 2-4 high-quality, diverse external
    resources to each module. Prefer free, concise, and beginner-friendly links
    (articles, videos, docs) and avoid duplicates across modules.

    Provide resources as a short list per module so tutor, quiz, and export
    agents can reuse them directly.
    """


def make_tutor_instruction() -> str:
    return """
    You are a friendly tutor agent.

    Given:
    - The user's current topic and context
    - A plan from the planner agent
    - Curated resources from the resource curator agent

    Your job is to:
    1. Explain the topic concisely
    2. Give 1-3 small practice prompts or mini-exercises
    3. Weave in the curated resources naturally (no tool calls)
    4. Prepare the user for a short quiz on this topic.

    Use simple language and concrete examples.
    """


def make_quiz_instruction() -> str:
    return """
    You are a quiz generator agent.

    For a given topic, create 3-5 short questions that check understanding.
    Mix multiple-choice and short-answer questions.

    Output your quiz as markdown with numbered questions and (for now)
    include the correct answers at the bottom under an "Answers" heading.

    Keep quizzes short and focused.
    """


def make_coach_instruction() -> str:
    return """
    You are the Pathfinder Learning Coach.

    You coordinate other agents to help the user learn effectively.

    When the user provides a goal, you should:
    1. Call the planner_agent sub-agent to create a plan.
    2. Call resource_curator_agent to align external links to each module.
    3. For the current session, call tutor_agent to teach the first topic using the curated resources.
    4. Call quiz_agent to generate short practice.
    5. Summarize what was covered and suggest what to do next.
    6. If the user asks for a report or wants to export their progress,
       call the export_agent sub-agent to generate an HTML report.

    Use a warm, encouraging tone.
    Do not show raw tool call traces; focus on a clean user experience.
    """


print("âœ… Agent instruction helpers ready.")





def make_export_instruction() -> str:
    return """
    You are an expert Course Content Generator Agent.

    Your ONLY job is to call the 'export_course_as_html' tool.
    
    CRITICAL RULES:
    1. DO NOT output any text to the user
    2. DO NOT explain what you're doing
    3. DO NOT use markdown formatting in your response
    4. IMMEDIATELY call the export_course_as_html tool with structured data
    
    When you receive a learning plan:
    1. Parse it into logical modules (typically 4-5 modules)
    2. For EACH module, generate:
       - title: A clear module name
       - explanation: 2-3 educational paragraphs explaining the topic
       - exercises: A list of 3-5 concrete practice exercises
       - resources: A list of 2-3 URLs to learning resources (use realistic URLs like:
         * https://docs.python.org/3/tutorial/
         * https://realpython.com/
         * https://www.w3schools.com/python/)
       - quiz: A list of 3 quiz questions, each with:
         * question: The question text
         * options: A list of 4 FULL TEXT answer options (NOT letters like A,B,C,D)
         * answer: The correct answer (must match one of the options exactly)
    
    3. Call export_course_as_html with:
       - course_title: A descriptive course title
       - modules: Your list of module dictionaries
       - output_filename: MUST be 'pathfinder_course.html'
    
    EXAMPLE of quiz format (use full text, not letters):
    quiz: [
        {
            "question": "What is Python?",
            "options": ["A programming language", "A snake", "A web browser", "A database"],
            "answer": "A programming language"
        }
    ]
    
    DO NOT DO ANY OF THE FOLLOWING:
    - Do not output text before calling the tool
    - Do not use Unicode emojis in your tool call
    - Do not ask the user for information
    - Do not explain your actions
    
    Call the tool IMMEDIATELY and ONLY.
    """

print("âœ… Export agent instruction helper ready (V3 - Fixed).")




# Shared model instance
base_model = Gemini(model=GEMINI_MODEL_NAME)




# Export Agent: generates detailed HTML courses
export_agent = LlmAgent(
    model=base_model,
    name="export_agent",
    description="Generates detailed HTML courses with explanations, exercises, and quizzes.",
    instruction=make_export_instruction(),
    tools=[export_course_as_html],
)

print("âœ… Export agent created (V2).")





# Planner Agent: turns goals into weekly plans
planner_agent = LlmAgent(
    model=base_model,
    name="planner_agent",
    description="Creates structured study plans based on a learning goal.",
    instruction=make_planner_instruction(),
)

# Resource Curator Agent: aligns external links to modules
resource_curator_agent = LlmAgent(
    model=base_model,
    name="resource_curator_agent",
    description="Finds high-quality resources mapped to each module in the plan.",
    instruction=make_resource_instruction(),
)

# Tutor Agent: explains topics and suggests practice
tutor_agent = LlmAgent(
    model=base_model,
    name="tutor_agent",
    description="Explains topics, gives small practice prompts, and uses curated resources.",
    instruction=make_tutor_instruction(),
)

# Quiz Agent: generates short quizzes
quiz_agent = LlmAgent(
    model=base_model,
    name="quiz_agent",
    description="Generates short quizzes to test understanding of a topic.",
    instruction=make_quiz_instruction(),
)

# Top-level Learning Coach Agent: orchestrates the others via sub_agents
learning_coach_agent = LlmAgent(
    model=base_model,
    name="learning_coach_agent",
    description=(
        "High-level learning coach that coordinates planner, resource curator, "
        "tutor, quiz, and export agents for self-directed learning."
    ),
    instruction=make_coach_instruction(),
    # Register sub-agents so the coach can delegate work
    sub_agents=[planner_agent, resource_curator_agent, tutor_agent, quiz_agent, export_agent],
    tools=[record_session_summary],
)

print("âœ… Agents created: planner_agent, resource_curator_agent, tutor_agent, quiz_agent, export_agent, learning_coach_agent")




    async def run_learning_session(
        user_goal: str,
        user_id: str = "demo_user",
        app_name: str = "pathfinder_app",
    ) -> None:
        """Run a single learning session with the coach agent.

        Args:
            user_goal: High-level learning goal, e.g. "Learn Python for data analysis in 4 weeks"
            user_id: Identifier for the learner (for sessions & memory)
            app_name: Application name for the session service
        """
        # Set up session management
        session_service = InMemorySessionService()
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Create a new session (required before running the agent)
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        runner = Runner(
            agent=learning_coach_agent,
            app_name=app_name,
            session_service=session_service,
        )

        # Wrap the user goal as Content for ADK / genai
        test_content = types.Content(
            role="user",
            parts=[types.Part(text=user_goal)],
        )

        print(f"ğŸ�¯ User goal: {user_goal}\n")

        # Stream the response
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=test_content
        ):
            if event.is_final_response() and event.content:
                print("ğŸ¤– Pathfinder Coach:")
                for part in event.content.parts:
                    if hasattr(part, "text"):
                        print(part.text)

        print("\nâœ… Session finished.")




# Unified Course Generation using planner_agent + export_agent

async def generate_course_with_agent(
    user_goal: str = "I want to learn DSA for FAANG interview in 4 weeks"
) -> None:
    app_name = "course_gen_app"
    user_id = "demo_user"
    session_service = InMemorySessionService()

    # --- Step 1: Get a structured plan from the planner_agent ---
    plan_session_id = f"course_plan_session_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=plan_session_id
    )

    plan_runner = Runner(
        agent=planner_agent,
        app_name=app_name,
        session_service=session_service,
    )

    plan_prompt = (
        f"Create a clear weekly learning plan for this goal:\n\n"
        f"{user_goal}\n\n"
        "Return only the plan as structured text (weeks + bullet points). "
        "Do NOT quiz, tutor, or exportâ€”just give the plan."
    )

    plan_content = types.Content(
        role="user",
        parts=[types.Part(text=plan_prompt)],
    )

    plan_text_parts: List[str] = []

    async for event in plan_runner.run_async(
        user_id=user_id,
        session_id=plan_session_id,
        new_message=plan_content,
    ):
        if event.is_final_response() and event.content:
            if event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        plan_text_parts.append(part.text)

    plan_text = "\n".join(plan_text_parts).strip()
    print("\ud83d\udcda Generated plan:\n")
    print(plan_text)
    print("\n")

    # --- Step 2: Ask export_agent to turn that plan into HTML (tool call) ---
    export_session_id = f"course_export_session_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=export_session_id
    )

    export_runner = Runner(
        agent=export_agent,
        app_name=app_name,
        session_service=session_service,
    )

    export_prompt = (
        "You are the Course Content Generator Agent.\n\n"
        "Using the learning plan below, generate a full course and then call the "
        "`export_course_as_html` tool exactly once with:\n"
        "- course_title: a short, descriptive title\n"
        "- modules: list of module dicts as described in your instructions\n"
        "- output_filename: 'pathfinder_course.html'\n\n"
        "Here is the learning plan:\n"
        f"{plan_text}"
    )

    export_content = types.Content(
        role="user",
        parts=[types.Part(text=export_prompt)],
    )

    print(f"\u25b6\ufe0f Starting course generation for: {user_goal}...")

    async for event in export_runner.run_async(
        user_id=user_id,
        session_id=export_session_id,
        new_message=export_content,
    ):
        if event.is_final_response() and event.content:
            print("\u2705 Course generation finished.")
            # (Optional) Print any final assistant text:
            if event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        safe_text = part.text.encode("utf-8", "replace").decode("utf-8")
                        print(safe_text)

# Execute the generation with a sample goal
goal = "I want to learn DSA for FAANG interview in 4 weeks, studying 5 hours per week."
await generate_course_with_agent(goal)

# Read and print the generated HTML file to console as requested
try:
    with open("pathfinder_course.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        print("\n" + "="*40 + " GENERATED HTML CONTENT " + "="*40 + "\n")
        print(html_content)
        print("\n" + "="*40 + " END OF HTML CONTENT " + "="*40 + "\n")
except FileNotFoundError:
    print("\u26a0\ufe0f Warning: 'pathfinder_course.html' was not found. Did the agent call the tool?")

from IPython.display import IFrame

IFrame(src='./pathfinder_course.html', width=700, height=600)


