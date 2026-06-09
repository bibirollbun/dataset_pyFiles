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


# Authentication by using API key from google
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


#We need Generative AI library and ADK components - lets import them
from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# We will also add some standard tools that are already available in adk and Built in Code executor package
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


"""
Tools for AI agent to interact with CSV data and execute pandas code.
Includes both CPU and GPU-accelerated execution options.
"""

import io
import sys
import contextlib
import time

# Added the below to update the library path necessary for some of the drivers needed.
import subprocess
import os

# Set environment variable
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = f"/kaggle/input/llamacpp-sm75-complete-build/build/bin:{env.get('LD_LIBRARY_PATH', '')}"




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


print("âœ… Helper functions to see generated python code and executed response defined.")


# =============================================================================
# PERSISTENT EXECUTION ENVIRONMENT
# =============================================================================

# Flag to track if GPU acceleration has been initialized
_GPU_INITIALIZED = False

# Persistent namespace that carries over between executions
PERSISTENT_NAMESPACE = {}


def _initialize_gpu_pandas():
    """Initialize GPU-accelerated pandas once at module level."""
    global _GPU_INITIALIZED
    if not _GPU_INITIALIZED:
        try:
            import cudf.pandas
            cudf.pandas.install()
            _GPU_INITIALIZED = True
            print("[GPU Acceleration] cudf.pandas initialized successfully")
        except Exception as e:
            print(f"[GPU Acceleration] Warning: Failed to initialize cudf.pandas: {e}")
            print("[GPU Acceleration] Falling back to CPU mode")


def _initialize_namespace():
    """Initialize or reset the persistent namespace with pandas."""
    global PERSISTENT_NAMESPACE
    # Import pandas AFTER cudf.pandas.install() has been called
    import pandas as pd
    import matplotlib.pyplot as plt
    PERSISTENT_NAMESPACE = {'pd': pd, 'plt': plt}
    print("Persistent namespace set with pandas.")


def reset_execution_environment():
    """Reset the persistent execution environment, clearing all variables."""
    global _GPU_INITIALIZED
    _initialize_namespace()
    return {
        "success": True,
        "message": "Execution environment reset. All variables cleared.",
        "gpu_mode": _GPU_INITIALIZED
    }

print("âœ… Persistent Execution Environment is set.")


# now we will initialize gpu enabled pandas libraries
_initialize_gpu_pandas()


# Initializing the persistent namespace
_initialize_namespace()


# =============================================================================
# TOOL FUNCTION - Get CSV Header
# =============================================================================

def get_csv_headers(file_path: str) -> dict:
    """
    Get the column headers from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary with columns list and count
    """
    try:
        import pandas as pd
        df_header = pd.read_csv(file_path, nrows=0)
        columns_list = df_header.columns.tolist()
        return {
            "success": True,
            "file_path": file_path,
            "columns": columns_list,
            "column_count": len(columns_list)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }



def _get_dataframe_info(namespace: dict) -> dict:
    """
    Extract information about pandas DataFrames in the namespace.

    Args:
        namespace: The execution namespace to inspect

    Returns:
        Dictionary mapping variable names to dataframe metadata
    """
    import pandas as pd
    dataframe_info = {}

    for var_name, var_value in namespace.items():
        # Skip built-in items and modules
        if var_name.startswith('_') or var_name == 'pd':
            continue

        # Check if it's a DataFrame
        if isinstance(var_value, pd.DataFrame):
            try:
                dataframe_info[var_name] = {
                    "shape": var_value.shape,
                    "columns": var_value.columns.tolist()[:10],  # Limit to first 10 columns
                    "dtypes": var_value.dtypes.astype(str).to_dict() if len(var_value.columns) <= 20 else "too_many_columns",
                    "memory_usage_mb": round(var_value.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            except Exception:
                # In case of any error getting metadata
                dataframe_info[var_name] = {
                    "shape": var_value.shape,
                    "columns": "error_getting_columns"
                }

    return dataframe_info


def _compute_dataframe_changes(before: dict, after: dict) -> dict:
    """
    Compute what dataframes were added, modified, or removed.

    Args:
        before: Dataframe info before execution
        after: Dataframe info after execution

    Returns:
        Dictionary with added, modified, and removed dataframe names
    """
    before_names = set(before.keys())
    after_names = set(after.keys())

    added = list(after_names - before_names)
    removed = list(before_names - after_names)

    # Check for modifications (shape change)
    modified = []
    for name in before_names & after_names:
        if before[name].get("shape") != after[name].get("shape"):
            modified.append(name)

    return {
        "added": added,
        "modified": modified,
        "removed": removed
    }


#lets test get_csv_headers
resp_dict = get_csv_headers("/kaggle/input/agencyperformance/finalapi.csv")
# Printing the column headers
print(resp_dict)


# Whats in the persistent namespace dataframe
persistent_df=_get_dataframe_info(PERSISTENT_NAMESPACE)
print(persistent_df)


def execute_python_code(code: str, use_gpu: bool = True, verbose: bool = True) -> dict:
    """
    Execute Python code using pandas with GPU acceleration.

    Variables persist between executions in the PERSISTENT_NAMESPACE,
    allowing dataframes to be reused across multiple calls.

    Note: GPU acceleration via cudf.pandas is initialized at module load time
    (before pandas import) and applies to all executions if available.

    Args:
        code: Python code string to execute
        use_gpu: Kept for API compatibility. GPU is initialized at module load.
        verbose: If True, print execution details and output. Default True.

    Returns:
        Dictionary with execution results, timing, dataframe tracking, and any errors
    """
    global PERSISTENT_NAMESPACE, _GPU_INITIALIZED

    mode = "gpu_accelerated" if _GPU_INITIALIZED else "cpu"

    if verbose:
        print(f"\n[Executing Python Code - {'GPU Accelerated' if _GPU_INITIALIZED else 'CPU'} Mode]")
        print("-" * 60)
        print(code)
        print("-" * 60)

    try:
        start_time = time.time()

        # Track dataframes before execution
        dataframes_before = _get_dataframe_info(PERSISTENT_NAMESPACE)

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, PERSISTENT_NAMESPACE)

        end_time = time.time()
        execution_time = end_time - start_time

        # Track dataframes after execution
        dataframes_after = _get_dataframe_info(PERSISTENT_NAMESPACE)

        # Determine what changed
        dataframe_changes = _compute_dataframe_changes(dataframes_before, dataframes_after)

        # Get captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Print the output so user can see it (only if verbose)
        if verbose:
            if stdout_output:
                print(stdout_output, end='')
            if stderr_output:
                print(stderr_output, end='', file=sys.stderr)

        return {
            "success": True,
            "mode": mode,
            "execution_time_seconds": round(execution_time, 4),
            "stdout": stdout_output,
            "stderr": stderr_output,
            "dataframes": dataframes_after,
            "dataframe_changes": dataframe_changes,
            "message": f"Code executed successfully on {'GPU' if _GPU_INITIALIZED else 'CPU'} in {execution_time:.4f} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "mode": mode,
            "error": str(e),
            "error_type": type(e).__name__
        }


# We will now test the execute python code function
code_result=execute_python_code("import pandas as pd\n\n# Read the CSV file\nfile_path = '/kaggle/input/agencyperformance/finalapi.csv'\ndf = pd.read_csv(file_path)\n\n# Print column names\nprint(df.columns)")
print(code_result)


# Lets build the helper functions that we need.
# we will create session if its not an active session
# we will get queries that could be more than 1 question at a time, in that case lets store them in a list
# and send one query at a time to LLM

# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


# To ensure reliability for our calls we may need to have retries configured. Your calls get failures 
# due to many factors - rate limits, temporary unavailability and etc.
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Lets build state. This will help us when we converse with our data
# We will also setup our agent that we will use in our session state.

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"


# Step 1: Create the LLM Agent
root_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="text_chat_bot",
    description="A text chatbot",  # Description of the agent's purpose
)

# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")



## Temporary code to validate if the above code is working - Kamal 
# Run a conversation with two queries in the same session
# Notice: Both queries are part of the SAME session, so context is maintained
await run_session(
    runner,
    [
        "Hi, I am Kamal! What is the capital of Tamil Nadu?",
        "Hello! What is my name?",  # This time, the agent should remember!
    ],
    "stateful-agentic-session",
)


# Lets build state. This will help us when we converse with our data
# We will also setup our agent that we will use in our session state.

APP_NAME = "Data Dialogue"  # Application
USER_ID = "Kamal"  # User
SESSION = "KamalSession"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"


# Step 1: Create the LLM Agent
dialogue_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="dialogue_agent",
    #Instruction to the LLM
    instruction="""You are a data analyst expert that ONLY responds with Python code.
    You are forbidden from providing any text, explanations, or conversational responses.
    Your task is to take the request for data analysis and write complete, executable Python code or call provided tools.
    **RULES:**
    1. Your output MUST be ONLY a executable Python code or call provided tools.
    2. Do NOT assume variables exist unless shown in the execution environment state.
    3. When asked to 'read' or 'load' a file, write the full code including 'import pandas as pd' and 'pd.read_csv()'. 
    4. Use print() to show results.
    5. Preserve exact case of data values. Don't change 'dialogue' to 'Dialogue'.
    6. Now answer user's request:
    
    Failure to follow these rules will result in an error.
    """, 
    #code_executor=BuiltInCodeExecutor(),  # Use the built-in Code Executor Tool. This gives the agent code execution capabilities
    tools=[execute_python_code],# Executor code written by me instead of using the in-built tool. 
)

# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=dialogue_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")



# Define a Dialogue Agent Runner - Not needed - Kamal
#dialogue_runner = InMemoryRunner(agent=dialogue_agent)


# Test the Data Dialogue agent
## Temporary code to validate if the above code is working - Kamal 
# Run a conversation with two queries in the same session
# Notice: Both queries are part of the SAME session, so context is maintained
await run_session(
    runner,
    [
        "Read /kaggle/input/agencyperformance/finalapi.csv",
        "What are the column names", 
    ],
    "stateful-agentic-session",
)
#response = await dialogue_runner.run_debug(
#    "Read /kaggle/input/agencyperformance/finalapi.csv what are the column names?"
#)


await run_session(
    runner,
    [
        "Read /kaggle/input/agencyperformance/finalapi.csv and list all the products that the agents sell",
        "Read /kaggle/input/agencyperformance/finalapi.csv and specify the agent that sells most of the products",
    ],
    "stateful-agentic-session",
)

