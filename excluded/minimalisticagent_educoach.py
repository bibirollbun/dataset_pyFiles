# Install Google GenAI SDK
!pip install google-genai


from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner,Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import AgentTool, google_search, load_memory, preload_memory, FunctionTool
from google.genai import types
from kaggle_secrets import UserSecretsClient

import os
import json
from datetime import datetime

print("âœ… ADK components imported successfully.")


# Retrieve API key from Kaggle secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Tool 1: Built in Google Search (for PlanningAgent and TeacherAgent)

# Tool 2: Execute Python Code (for TeacherAgent)
def execute_python_code(code: str) -> str:
    """
    Executes Python code safely and returns the output.
    
    Args:
        code: Python code to execute
        
    Returns:
        The output of the executed code or error message
    """
    try:
        # Create a safe execution environment
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        
        # Capture any print statements or return values
        if 'result' in local_vars:
            return str(local_vars['result'])
        else:
            return "Code executed successfully. No explicit return value."
    except Exception as e:
        return f"Error executing code: {str(e)}"


# Tool 3: Grade Quiz (for EvaluationAgent)
def grade_quiz(student_answers: str, correct_answers: str) -> str:
    """
    Grades a quiz by comparing student answers with correct answers.
    
    Args:
        student_answers: JSON string of student answers
        correct_answers: JSON string of correct answers
        
    Returns:
        A string containing the grade and feedback
    """
    try:
        student_dict = json.loads(student_answers)
        correct_dict = json.loads(correct_answers)
        
        total_questions = len(correct_dict)
        correct_count = 0
        feedback = []
        
        for question_id, correct_answer in correct_dict.items():
            student_answer = student_dict.get(question_id, "No answer provided")
            if str(student_answer).strip().lower() == str(correct_answer).strip().lower():
                correct_count += 1
                feedback.append(f"{question_id}: Correct!")
            else:
                feedback.append(f"{question_id}: Incorrect. Your answer: {student_answer}, Correct answer: {correct_answer}")
        
        score = (correct_count / total_questions) * 100
        result = f"Score: {score:.1f}% ({correct_count}/{total_questions} correct)\n\n"
        result += "\n".join(feedback)
        
        return result
    except Exception as e:
        return f"Error grading quiz: {str(e)}"


print("Custom tools defined successfully!")


# Agent 1: PlanningAgent
# Creates personalized learning plans based on student requirements

planning_agent = LlmAgent(
    name="PlanningAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a PlanningAgent responsible for creating personalized learning plans for students.

Your responsibilities:
1. Analyze the student's learning request (topic, duration, student ID)
2. Break down the topic into manageable daily lessons
3. Create a structured learning plan with clear objectives for each day
4. Use the google_search tool to find relevant educational resources
5. Ensure the plan is realistic and achievable within the given timeframe

Output format:
Provide a detailed learning plan with:
- Student ID
- Topic
- Duration in days
- Daily lessons with titles, objectives, and recommended resources

Be specific and actionable in your planning.
""",
    tools=[google_search,preload_memory],
    after_agent_callback=auto_save_to_memory,
)

print("PlanningAgent created successfully!")


# Agent 2: TeacherAgent
# Delivers educational content and explanations

teacher_agent = LlmAgent(
    name="TeacherAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a TeacherAgent responsible for delivering educational content to students.

Your responsibilities:
1. Explain concepts clearly and concisely for beginners
2. Provide practical examples and demonstrations
3. Use the google_search tool to find additional learning resources
4. Use the execute_python_code tool to demonstrate code examples
5. Adapt your teaching style to the student's level

Teaching guidelines:
- Start with simple explanations
- Use real-world analogies
- Provide code examples that can be executed
- Encourage hands-on practice
- Be patient and supportive

Deliver comprehensive lessons that students can follow and practice.
""",
    tools=[google_search, execute_python_code, preload_memory],
    after_agent_callback=auto_save_to_memory,
    
)

print("TeacherAgent created successfully!")


# Agent 3: QuizAgent
# Generates assessments to test student understanding

quiz_agent = LlmAgent(
    name="QuizAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a QuizAgent responsible for creating assessments to test student understanding.

Your responsibilities:
1. Generate relevant quiz questions based on the topics covered
2. Create questions at appropriate difficulty levels
3. Include multiple choice, true/false, and short answer questions
4. Provide clear question statements
5. Store correct answers for evaluation

Output format:
Provide a structured quiz with:
- Quiz title and topic
- 3-5 questions with clear instructions
- Question types (multiple choice, true/false, short answer)
- Options for multiple choice questions
- Correct answers in JSON format at the end: {\"q1\": \"answer\", \"q2\": \"answer\"}

Make questions clear, fair, and aligned with the lesson content.
""",
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory,
)

print("QuizAgent created successfully!")


# Agent 4: EvaluationAgent
# Evaluates student performance and provides feedback

evaluation_agent = LlmAgent(
    name="EvaluationAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are an EvaluationAgent responsible for evaluating student performance and providing constructive feedback.

Your responsibilities:
1. Use the grade_quiz tool to score student assessments
2. Analyze student performance and identify strengths and weaknesses
3. Provide constructive feedback and encouragement
4. Suggest areas for improvement and additional practice
5. Track progress over time

Evaluation guidelines:
- Be encouraging and positive
- Highlight what the student did well
- Gently point out areas needing improvement
- Provide specific recommendations
- Celebrate progress and milestones

First use the grade_quiz tool, then provide detailed feedback based on the results.
""",
    tools=[grade_quiz,preload_memory],
    after_agent_callback=auto_save_to_memory,
)

print("EvaluationAgent created successfully!")


memory_service = InMemoryMemoryService() # ADK's built-in Memory Service for development and testing
session_service = InMemorySessionService()

async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("âœ… Callback created.")


# Agent 5: OrchestratorAgent
 #Coordinates the workflow by calling other agents one after another
orchestrator_agent_seq = SequentialAgent(
     name="OrchestratorAgent",
     sub_agents=[planning_agent, teacher_agent, quiz_agent, evaluation_agent]
)

print("âœ… Sequential Agent created.")
# orchestrator_agent = LlmAgent(
#     name="OrchestratorAgent",
#     model="gemini-2.5-flash-lite",
#     instruction="""
# You are an OrchestratorAgent responsible for coordinating the educational coaching workflow.

# Your responsibilities:
# 1. Parse the learning request to extract: student_id, topic, and duration
# 2. Call the PlanningAgent to create a learning plan
# 3. Call the TeacherAgent to deliver the first lesson
# 4. Call the QuizAgent to generate an assessment
# 5. Call the EvaluationAgent to grade and provide feedback
# 6. Coordinate the sequential flow of information between agents

# Workflow:
# Step 1: Analyze the request and extract student_id, topic, duration
# Step 2: Request PlanningAgent to create a plan
# Step 3: Request TeacherAgent to teach Day 1 lesson
# Step 4: Request QuizAgent to create a quiz for Day 1
# Step 5: Request EvaluationAgent to evaluate (with simulated student answers)

# You will receive outputs from each agent and pass relevant information to the next agent.
# Memory context will be provided in the prompt.

# Output: Provide a summary of the entire coaching session including all agent responses.
# """,
#     tools=[preload_memory],
#     after_agent_callback=auto_save_to_memory,
# )

print("OrchestratorAgent created successfully!")



# Create a runner for the auto-save agent
# This connects our automated agent to the session and memory services
APP_NAME = "EduCoach"
coach_runner = Runner(
    agent=orchestrator_agent,  # Use the agent with callback + preload_memory
    app_name=APP_NAME,
    session_service=session_service,  # Same services from Section 3
    memory_service=memory_service,
)

print("âœ… Runner created.")



async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("âœ… Helper functions defined.")


# Test 1: Ask agent to teach Python
# The callback will automatically save this to memory when the turn completes
# Example learning request

USER_ID = "Liza"
SESSION_ID = "python-day-1"

learning_request = "Learn Python basics in 3 days for student_001"
await run_session(
    coach_runner,
    learning_request,
    SESSION_ID,
)


# Test 2: Testing the session memory from python-day-1
# The callback will automatically save this to memory when the turn completes
# Example learning request

USER_ID = "Liza"
SESSION_ID = "python-day-1"

learning_request = "what was the score of my previous quiz test?"
await run_session(
    coach_runner,
    learning_request,
    SESSION_ID,
)


