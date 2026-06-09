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
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Building a simple agent that gives advice based on BMI input

# Define a custom tool to give suggestions
def bmi_advice_tool(bmi: float) -> str:
    """Return health advice based on BMI."""
    if bmi < 18.5:
        return "Underweight: Increase calorie intake, focus on protein, and do strength training."
    elif 18.5 <= bmi < 24.9:
        return "Normal: Maintain your current lifestyle with balanced diet and regular exercise."
    elif 24.9 <= bmi < 29.9:
        return "Overweight: Control calorie intake, increase physical activity, and avoid sugary food."
    else:
        return "Obese: Follow a diet plan, do regular exercise, and consult a health professional."

# Define the agent
bmi_agent = LlmAgent(
    name="bmi_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that gives suggestions based on BMI input.",
    instruction="You are a helpful assistant. Ask the user for their BMI and return health advice.",
    tools=[bmi_advice_tool],  # using our custom tool
)

print("âœ… BMI suggestion agent defined.")



# Step 1: Define the BMI calculator agent (similar to calculation_agent)
bmi_calculation_agent = LlmAgent(
    name="bmi_calculation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are a specialized BMI advisor that ONLY responds with Python code. You are forbidden from providing any text or explanations. 

Your task is to take a BMI value as input and generate Python code that prints the BMI category and recommendation. 
Follow these rules strictly:
1. Output ONLY a Python code block.
2. Do NOT write text before or after the code.
3. Python code MUST print the result.
4. Do NOT calculate the advice yourself, only generate code.
""",
    code_executor=BuiltInCodeExecutor(),
)

print("âœ… BMI calculation agent defined.")



# Step 2: Define the main BMI suggestion agent (orchestration agent)
bmi_suggestion_agent = LlmAgent(
    name="bmi_suggestion_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are a smart BMI suggestion agent. For any BMI advice request:

1. Get the BMI value from the user input.
2. Call the bmi_calculation_agent to generate Python code that prints the BMI category and advice.
3. Execute the code using the code_executor.
4. Return the printed result to the user.
""",
    tools=[
        AgentTool(agent=bmi_calculation_agent)
    ],
)

print("âœ… BMI suggestion agent created")
print("ðŸŽ¯ Takes BMI input and delegates advice generation to BMI calculation agent")



# Defining a runner for the BMI suggestion agent
bmi_runner = InMemoryRunner(agent=bmi_suggestion_agent)

print("âœ… Runner for BMI suggestion agent defined")



# Testing the agent
response = await bmi_runner.run_debug(
    "17"
)

