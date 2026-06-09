!pip install google-adk google-genai numpy pandas --quiet
print("âœ… Installing dependencies done.")


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("âœ… Checking environment done.")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner, InMemoryRunner
from google.adk.tools import FunctionTool, google_search
from google import genai
from google.genai import types

from kaggle_secrets import UserSecretsClient

import json

import logging

print("âœ… Importing libraries done.")


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


APP_NAME                        = 'ConformanceTesterAIAgent'
USER_ID                         = 'ConformanceTesterEngineer'
SESSION_ID                      = 'conformance_tester_session_01'
MODEL                           = 'gemini-2.5-flash'
#MODEL                           = 'model="gemini-3-pro-preview"'
AGENT_NAME                      = 'ConformanceTesterAgent'
AGENT_DESCRIPTION               = 'A conformance tester agent.'
METHODOLOGY_AGENT_NAME          = 'MethodologyAgent'
METHODOLOGY_AGENT_DESCRIPTION   = 'Acquire the methodology to create conformance test objectives from an unknown standard.'
EDITING_RULES_AGENT_NAME        = 'EditingRulesAgent'
EDITING_RULES_AGENT_DESCRIPTION = 'Acquire the editing rukes in order to format output ducments at the end of the process.'
PARALLEL_LEARNING_AGENT_NAME    = 'ParallelLearningAgent'
PARALLEL_LEARNING_AGENT_DESCRIPTION = 'Parallelisation of learning agents.'
LEARNING_AGGREG_AGENT           = 'LearningAggregAgent'
PICS_AGENT_NAME                 = 'PicsAgent'
PICS_AGENT_DESCRIPTION          = 'Extract the PICS from the unknown base standard.'
TO_AGENT_NAME                   = 'ToAgent'
TO_AGENT_DESCRIPTION            = 'Extract the test objectives from the unknown base standard.'
PARALLEL_EXAMIN_AGENT_NAME      = 'ParallelStandardExamAgent'
PARALLEL_EXAMIN_AGENT_DESCRIPTION = 'Parallelisation of PICS and test objectives extraction from the unknown base standard.'
CONSOLIDATE_AGENT_NAME          = 'ConsolidateAgent'
FORMATTING_AGENT_NAME           = 'FormattingAgent'
FORMATTING_DESCRIPTION          = 'Formatting output documents according to the ETSI rules'
EVALUATOR_AGENT_NAME            = 'EvaluatorAgent'
EVALUATOR_AGENT_DESCRIPTION     = 'Evaluate the produced documents'
REVIEWER_AGENT_NAME             = 'ReviewerAgent'
REVIEWER_AGENT_DESCRIPTION      = 'Review the produced documents'
DOCUMENTS_REVIEW_LOOP_AGENT_NAME = 'DocumentsReviewLoop'
DOCUMENTS_REVIEW_LOOP_AGENT_DESCRIPTION = 'Reviewer agent'
TD_AGENT_NAME                   = 'TdAgent'
TD_AGENT_DESCRIPTION            = 'Develop the test description for each test objective.'

MAX_REVIEWING_ITERATION         = 2 # To prevent infinite loops

conformance_tester_agent         = None # This is the root agent.
methodology_agent                = None # This agent learns the ISO/IEC 9646-1 to ISO/IEC 9646-7 methodology.
editing_rules_agent              = None # This agent learns the ETSI editing rules.
parallel_learning_agent          = None # This agent manages the parallelized sub-agent for conformance test learning phase.
learning_aggreg_agent            = None
pics_agent                       = None # This agent extracts the PICS from the unknown base standard. It runs in parallel with to_agent.
to_agent                         = None # This agent extracts the test objectives from the unknown base standard. It runs in parallel with pics_agent.
parallel_standard_analysis_agent = None # This agent manages the parallelized sub-agent.
consolidate_agent                = None # This agent enhance PICS requirement in the test objectives description.
formatting_agent                 = None # This agent formats the output document according to ETSI Editing rules.
evaluator_agent                  = None # This agent evaluates the output produced during the reviewing phase
reviewer_agent                   = None # This agent does the review of the output produced
documents_review_loop_agent      = None # This is the rviewing loop agent
td_agent                         = None # This agent develops the test description for each test objectives.

print("âœ… Global data structures done.")


DOMAIN                     = 'ETSI MEC'
METHODOLOGY_INSTRUCTIONS   = """You are an expert in standardization. You are also an expert in conformance testing as defined in ISO/IEC 9646-1 to ISO/IEC 9646-7.
As an conformance tester, you are asked to acquire the methodology to extract:
1. the Protocol Implementation Conformance Statement (PICS)
2. the test objectives summary description
3. the references to the base standard, indicating the clause number, paragraph and sub-paragraph.
Before to learn from ISO/IEC 9646-1 to ISO/IEC 9646-7, you can examine ETSI approach of conformance testing at this URI: https://portal.etsi.org/Services/Centre-for-Testing-Interoperability/ETSI-Approach/Conformance
You will use what you learned during the extraction of the test objectives and the PICS.
Note: As an argentic agent, you help a human engineer to generate the complete PICS and Test Objectives documents as meticulously yo can.
"""
EDITING_RULES_INSTRUCTIONS = """You are asked learn the ETSI editing rules as defined here: https://portal.etsi.org/Services/editHelp/How-to-start/ETSI-Drafting-Rules.
You will use what you learned during the formatting of the output documents."""
LEARNING_AGGREG_INSTRUCTIONS = """Combine these two knowledges into a single one:

**Conformance testing methodology:**
{methodology}

**Editing rules:**
{editing_rules}

You will use what you learned during later.
"""
PICS_INSTRUCTIONS          = """Using the methodology you learned, extract the PICS from the unknown standard. 
The output will be a markdown formatted document and the name of this document is 'pics.md'. It will be stored in my working directory.
"""
TO_INSTRUCTIONS            = """Using the methodology you learned, extract the test objectives from the unknown standard. 
The output will be a markdown formatted document and the name of this document is 'test_objective.md' It will be stored in my working directory.
"""
CONSOLIDATE_INSTRUCTIONS   = """Using the PICS document you generated, enhance the test objectives descriptions with the proper PICS requirements following the methodology you learned, i.e. ISO/IEC 9646."""
FORMATTING_INSTRUCTIONS    = """You are asked to format the generated files according to the ETSI Editing rules defined here: {editing_rules}."""
EVALUATOR_INSTRUCTIONS     = """As expert in standardization domain and in ETSI MEC domain, evaluate the output {formatting}.
If you think the documents you produced are completed and cover the whole unknown base standard (all the PICS are well described and references and all the clause and tables of the unknown base standard are covered by the test objectives), then you can stop the refinement loop you MUST respond with the exact phrase: "COMPLETED". 
Otherwise, redo a new run of refinement.
"""
REVIEWER_INSTRUCTIONS     = """You are a conformance tester reviewer. You have a Formatted Output and an Evaluation:
Formatted Output: {formatting}
Evaluation: {evaluation}

Your task is to analyze the evaluation.
- IF the evaluation is EXACTLY "COMPLETED", you MUST call the `terminate_review` function and nothing else.
- OTHERWISE, review the formatted output to fully incorporate the feedback from the evaluation.
"""
TD_INSTRUCTIONS           = """You are a conformance tester. You have a Formatted Output containing a stable draft of all the test objectives exctrated from the unknown base standard:
Formatted Output: {formatting}

Your task is to provide a detailled test description of each test objective. To help you in this task, remember what you learn in:

**Conformance testing methodology:**
{methodology}

**Editing rules:**
{editing_rules}

More specifically, refer to ISO/IEC 9646-3 to structure each test description.
"""

USER_PROMPT                = """You are an expert in ETSI MEC. You are asked to produce PICS and Test Objectives documents for the standard ETSI GS MEC 030 (https://www.etsi.org/deliver/etsi_gs/MEC/001_099/030/03.03.01_60/gs_MEC030v030301p.pdf)."""

print("âœ… Prompts definitions done.")


methodology_agent = LlmAgent(
    name=METHODOLOGY_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=METHODOLOGY_INSTRUCTIONS,
    tools=[google_search],
    output_key="methodology",
)

print("âœ… methodology_agent created.")


editing_rules_agent = LlmAgent(
    name=EDITING_RULES_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=EDITING_RULES_INSTRUCTIONS,
    description=EDITING_RULES_AGENT_DESCRIPTION,
    tools=[google_search],
    output_key="editing_rules",
)

print(f"âœ… {EDITING_RULES_AGENT_NAME} created.")


parallel_learning_agent = ParallelAgent(
    name=PARALLEL_LEARNING_AGENT_NAME,
    sub_agents=[methodology_agent, editing_rules_agent],
    description=PARALLEL_LEARNING_AGENT_DESCRIPTION
)
print(f"âœ… {PARALLEL_LEARNING_AGENT_NAME} created.")


learning_aggreg_agent = LlmAgent(
    name=LEARNING_AGGREG_AGENT,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction=LEARNING_AGGREG_INSTRUCTIONS,
    output_key="learning_aggreg",  # This will be the final output of the entire system.
)

print(f"âœ… {LEARNING_AGGREG_AGENT} created.")


pics_agent = LlmAgent(
    name=PICS_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=PICS_INSTRUCTIONS,
    tools=[google_search],
    output_key="pics",
)

print(f"âœ… {PICS_AGENT_NAME} created.")


to_agent = LlmAgent(
    name=TO_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=TO_INSTRUCTIONS,
    tools=[google_search],
    output_key="to",
)

print(f"âœ… {TO_AGENT_NAME} created.")


parallel_standard_analysis = ParallelAgent(
    name=PARALLEL_EXAMIN_AGENT_NAME,
    sub_agents=[pics_agent, to_agent],
    description=PARALLEL_EXAMIN_AGENT_DESCRIPTION
)
print(f"âœ… {PARALLEL_EXAMIN_AGENT_NAME} created.")


consolidate_agent = LlmAgent(
    name=CONSOLIDATE_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=CONSOLIDATE_INSTRUCTIONS,
    tools=[google_search],
    output_key="consolidate",
)

print(f"âœ… {TO_AGENT_NAME} created.")


formatting_agent = LlmAgent(
    name=FORMATTING_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=FORMATTING_INSTRUCTIONS,
    tools=[google_search],
    output_key="formatting",
)

print(f"âœ… {TO_AGENT_NAME} created.")


evaluator_agent = Agent(
    name=EVALUATOR_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=EVALUATOR_INSTRUCTIONS,
    description=EVALUATOR_AGENT_DESCRIPTION,
    output_key="evaluation",
)

print("âœ… EvaluatorAgent created.")


def terminate_review():
    """Call this function ONLY when the evaluation is 'COMPLETED', indicating the documents are in early draft state."""
    return {"status": "completed", "message": "the documents are in early draft state. Exiting reviewing loop."}

print("âœ… terminate_review function created.")


reviewer_agent = Agent(
    name=REVIEWER_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=REVIEWER_INSTRUCTIONS,
    description=REVIEWER_AGENT_DESCRIPTION,
    output_key="formatting",  # It overwrites the formatting outputs.
    tools=[
        FunctionTool(terminate_review)
    ],
)

print("âœ… ReviewerAgent created.")


documents_review_loop_agent = LoopAgent(
    name="DocumentsReviewLoop",
    sub_agents=[evaluator_agent, reviewer_agent],
    max_iterations=MAX_REVIEWING_ITERATION,  # Prevents infinite loops
)

print("âœ… DocumentsReviewLoop created.")


td_agent = LlmAgent(
    name=TD_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=TD_INSTRUCTIONS,
    tools=[google_search],
    output_key="td",
)

print(f"âœ… {TD_AGENT_NAME} created.")


conformance_tester_agent = SequentialAgent(
    name=AGENT_NAME,
    sub_agents=[parallel_learning_agent, learning_aggreg_agent, parallel_standard_analysis, consolidate_agent, formatting_agent, documents_review_loop_agent],
    description=AGENT_DESCRIPTION,
)

print(f"âœ… {AGENT_NAME} created.")


session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=conformance_tester_agent, app_name=APP_NAME, session_service=session_service)
response = await runner.run_debug(USER_PROMPT)


%%script echo skipping
# Comment the line above to execute this cell
!adk create ConformanceTesterAIAgent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%script echo skipping
# Comment the line above to execute this cell
%%writefile research-agent/ConformanceTesterAIAgent.py

#############################################################################
# Checking environment
#############################################################################

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("âœ… Checking environment done.")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#############################################################################
# Importing libraries
#############################################################################

from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner, InMemoryRunner
from google.adk.tools import FunctionTool, google_search
from google import genai
from google.genai import types

from kaggle_secrets import UserSecretsClient

import json

import logging

print("âœ… Importing libraries done.")

#############################################################################
# Setup and authentication
#############################################################################

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

#############################################################################
# Configure Retry Options
#############################################################################

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

#############################################################################
# Logging function
#############################################################################

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")

#############################################################################
# Global data structures
#############################################################################

APP_NAME                        = 'ConformanceTesterAIAgent'
USER_ID                         = 'ConformanceTesterEngineer'
SESSION_ID                      = 'conformance_tester_session_01'
MODEL                           = 'gemini-2.5-flash'
#MODEL                           = 'model="gemini-3-pro-preview"'
AGENT_NAME                      = 'ConformanceTesterAgent'
AGENT_DESCRIPTION               = 'A conformance tester agent.'
METHODOLOGY_AGENT_NAME          = 'MethodologyAgent'
METHODOLOGY_AGENT_DESCRIPTION   = 'Acquire the methodology to create conformance test objectives from an unknown standard.'
EDITING_RULES_AGENT_NAME        = 'EditingRulesAgent'
EDITING_RULES_AGENT_DESCRIPTION = 'Acquire the editing rukes in order to format output ducments at the end of the process.'
PARALLEL_LEARNING_AGENT_NAME    = 'ParallelLearningAgent'
PARALLEL_LEARNING_AGENT_DESCRIPTION = 'Parallelisation of learning agents.'
LEARNING_AGGREG_AGENT           = 'LearningAggregAgent'
PICS_AGENT_NAME                 = 'PicsAgent'
PICS_AGENT_DESCRIPTION          = 'Extract the PICS from the unknown base standard.'
TO_AGENT_NAME                   = 'ToAgent'
TO_AGENT_DESCRIPTION            = 'Extract the test objectives from the unknown base standard.'
PARALLEL_EXAMIN_AGENT_NAME      = 'ParallelStandardExamAgent'
PARALLEL_EXAMIN_AGENT_DESCRIPTION = 'Parallelisation of PICS and test objectives extraction from the unknown base standard.'
CONSOLIDATE_AGENT_NAME          = 'ConsolidateAgent'
FORMATTING_AGENT_NAME           = 'FormattingAgent'
FORMATTING_DESCRIPTION          = 'Formatting output documents according to the ETSI rules'
EVALUATOR_AGENT_NAME            = 'EvaluatorAgent'
EVALUATOR_AGENT_DESCRIPTION     = 'Evaluate the produced documents'
REVIEWER_AGENT_NAME             = 'ReviewerAgent'
REVIEWER_AGENT_DESCRIPTION      = 'Review the produced documents'
DOCUMENTS_REVIEW_LOOP_AGENT_NAME = 'DocumentsReviewLoop'
DOCUMENTS_REVIEW_LOOP_AGENT_DESCRIPTION = 'Reviewer agent'
TD_AGENT_NAME                   = 'TdAgent'
TD_AGENT_DESCRIPTION            = 'Develop the test description for each test objective.'

MAX_REVIEWING_ITERATION         = 2 # To prevent infinite loops

conformance_tester_agent         = None # This is the root agent.
methodology_agent                = None # This agent learns the ISO/IEC 9646-1 to ISO/IEC 9646-7 methodology.
editing_rules_agent              = None # This agent learns the ETSI editing rules.
parallel_learning_agent          = None # This agent manages the parallelized sub-agent for conformance test learning phase.
learning_aggreg_agent            = None
pics_agent                       = None # This agent extracts the PICS from the unknown base standard. It runs in parallel with to_agent.
to_agent                         = None # This agent extracts the test objectives from the unknown base standard. It runs in parallel with pics_agent.
parallel_standard_analysis_agent = None # This agent manages the parallelized sub-agent.
consolidate_agent                = None # This agent enhance PICS requirement in the test objectives description.
formatting_agent                 = None # This agent formats the output document according to ETSI Editing rules.
evaluator_agent                  = None # This agent evaluates the output produced during the reviewing phase
reviewer_agent                   = None # This agent does the review of the output produced
documents_review_loop_agent      = None # This is the rviewing loop agent
td_agent                         = None # This agent develops the test description for each test objectives.

print("âœ… Global data structures done.")

#############################################################################
# Prompts definitions
#############################################################################

DOMAIN                     = 'ETSI MEC'
METHODOLOGY_INSTRUCTIONS   = """You are an expert in standardization. You are also an expert in conformance testing as defined in ISO/IEC 9646-1 to ISO/IEC 9646-7.
As an conformance tester, you are asked to acquire the methodology to extract:
1. the Protocol Implementation Conformance Statement (PICS)
2. the test objectives summary description
3. the references to the base standard, indicating the clause number, paragraph and sub-paragraph.
Before to learn from ISO/IEC 9646-1 to ISO/IEC 9646-7, you can examine ETSI approach of conformance testing at this URI: https://portal.etsi.org/Services/Centre-for-Testing-Interoperability/ETSI-Approach/Conformance
You will use what you learned during the extraction of the test objectives and the PICS.
Note: As an argentic agent, you help a human engineer to generate the complete PICS and Test Objectives documents as meticulously yo can.
"""
EDITING_RULES_INSTRUCTIONS = """You are asked learn the ETSI editing rules as defined here: https://portal.etsi.org/Services/editHelp/How-to-start/ETSI-Drafting-Rules.
You will use what you learned during the formatting of the output documents."""
LEARNING_AGGREG_INSTRUCTIONS = """Combine these two knowledges into a single one:

**Conformance testing methodology:**
{methodology}

**Editing rules:**
{editing_rules}

You will use what you learned during later.
"""
PICS_INSTRUCTIONS          = """Using the methodology you learned, extract the PICS from the unknown standard. 
The output will be a markdown formatted document and the name of this document is 'pics.md'. It will be stored in my working directory.
"""
TO_INSTRUCTIONS            = """Using the methodology you learned, extract the test objectives from the unknown standard. 
The output will be a markdown formatted document and the name of this document is 'test_objective.md' It will be stored in my working directory.
"""
CONSOLIDATE_INSTRUCTIONS   = """Using the PICS document you generated, enhance the test objectives descriptions with the proper PICS requirements following the methodology you learned, i.e. ISO/IEC 9646."""
FORMATTING_INSTRUCTIONS    = """You are asked to format the generated files according to the ETSI Editing rules defined here: {editing_rules}."""
EVALUATOR_INSTRUCTIONS     = """As expert in standardization domain and in ETSI MEC domain, evaluate the output {formatting}.
If you think the documents you produced are completed and cover the whole unknown base standard (all the PICS are well described and references and all the clause and tables of the unknown base standard are covered by the test objectives), then you can stop the refinement loop you MUST respond with the exact phrase: "COMPLETED". 
Otherwise, redo a new run of refinement.
"""
REVIEWER_INSTRUCTIONS     = """You are a conformance tester reviewer. You have a Formatted Output and an Evaluation:
Formatted Output: {formatting}
Evaluation: {evaluation}

Your task is to analyze the evaluation.
- IF the evaluation is EXACTLY "COMPLETED", you MUST call the `terminate_review` function and nothing else.
- OTHERWISE, review the formatted output to fully incorporate the feedback from the evaluation.
"""
TD_INSTRUCTIONS           = """You are a conformance tester. You have a Formatted Output containing a stable draft of all the test objectives exctrated from the unknown base standard:
Formatted Output: {formatting}

Your task is to provide a detailled test description of each test objective. To help you in this task, remember what you learn in:

**Conformance testing methodology:**
{methodology}

**Editing rules:**
{editing_rules}

More specifically, refer to ISO/IEC 9646-3 to structure each test description.
"""

USER_PROMPT                = """You are an expert in ETSI MEC. You are asked to produce PICS and Test Objectives documents for the standard ETSI GS MEC 030 (https://www.etsi.org/deliver/etsi_gs/MEC/001_099/030/03.03.01_60/gs_MEC030v030301p.pdf)."""

print("âœ… Prompts definitions done.")

#############################################################################
# Methodoloy agent
#############################################################################

methodology_agent = LlmAgent(
    name=METHODOLOGY_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=METHODOLOGY_INSTRUCTIONS,
    tools=[google_search],
    output_key="methodology",
)

print("âœ… methodology_agent created.")

#############################################################################
# Learning editing rules
#############################################################################

editing_rules_agent = LlmAgent(
    name=EDITING_RULES_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=EDITING_RULES_INSTRUCTIONS,
    description=EDITING_RULES_AGENT_DESCRIPTION,
    tools=[google_search],
    output_key="editing_rules",
)

print(f"âœ… {EDITING_RULES_AGENT_NAME} created.")

#############################################################################
# Parallelization of learning agents
#############################################################################

parallel_learning_agent = ParallelAgent(
    name=PARALLEL_LEARNING_AGENT_NAME,
    sub_agents=[methodology_agent, editing_rules_agent],
)
print(f"âœ… {PARALLEL_LEARNING_AGENT_NAME} created.")

#############################################################################
# Aggregating the knowledges
#############################################################################

learning_aggreg_agent = LlmAgent(
    name=LEARNING_AGGREG_AGENT,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction=LEARNING_AGGREG_INSTRUCTIONS,
    output_key="learning_aggreg",  # This will be the final output of the entire system.
)

print(f"âœ… {LEARNING_AGGREG_AGENT} created.")

#############################################################################
# PICS extraction agent
#############################################################################

pics_agent = LlmAgent(
    name=PICS_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=PICS_INSTRUCTIONS,
    tools=[google_search],
    output_key="pics",
)

print(f"âœ… {PICS_AGENT_NAME} created.")

#############################################################################
# Test objectives extraction agent
#############################################################################

to_agent = LlmAgent(
    name=TO_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=TO_INSTRUCTIONS,
    tools=[google_search],
    output_key="to",
)

print(f"âœ… {TO_AGENT_NAME} created.")

#############################################################################
# Parallelization of extraction agents
#############################################################################

parallel_standard_analysis = ParallelAgent(
    name=PARALLEL_EXAMIN_AGENT_NAME,
    sub_agents=[pics_agent, to_agent],
    description=PARALLEL_EXAMIN_AGENT_DESCRIPTION
)
print(f"âœ… {PARALLEL_EXAMIN_AGENT_NAME} created.")

#############################################################################
# Consolidating PICS and Test Objective documents
#############################################################################

consolidate_agent = LlmAgent(
    name=CONSOLIDATE_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=CONSOLIDATE_INSTRUCTIONS,
    tools=[google_search],
    output_key="consolidate",
)

print(f"âœ… {TO_AGENT_NAME} created.")

#############################################################################
# Formatting output documents
#############################################################################

formatting_agent = LlmAgent(
    name=FORMATTING_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=FORMATTING_INSTRUCTIONS,
    tools=[google_search],
    output_key="formatting",
)

print(f"âœ… {TO_AGENT_NAME} created.")

#############################################################################
# Evaluator Agent
#############################################################################

evaluator_agent = Agent(
    name=EVALUATOR_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=EVALUATOR_INSTRUCTIONS,
    description=EVALUATOR_AGENT_DESCRIPTION,
    output_key="evaluation",
)

print("âœ… EvaluatorAgent created.")

#############################################################################
# Terminate review iterations
#############################################################################

def terminate_review():
    """Call this function ONLY when the evaluation is 'COMPLETED', indicating the documents are in early draft state."""
    return {"status": "completed", "message": "the documents are in early draft state. Exiting reviewing loop."}

print("âœ… terminate_review function created.")

#############################################################################
# Reviewer Agent
#############################################################################

reviewer_agent = Agent(
    name=REVIEWER_AGENT_NAME,
    model=Gemini(
        model=MODEL,
        retry_options=retry_config
    ),
    instruction=REVIEWER_INSTRUCTIONS,
    description=REVIEWER_AGENT_DESCRIPTION,
    output_key="formatting",  # It overwrites the formatting outputs.
    tools=[
        FunctionTool(terminate_review)
    ],
)

print("âœ… ReviewerAgent created.")

#############################################################################
# Documents review iteration Agent
#############################################################################

documents_review_loop_agent = LoopAgent(
    name="DocumentsReviewLoop",
    sub_agents=[evaluator_agent, reviewer_agent],
    max_iterations=MAX_REVIEWING_ITERATION,  # Prevents infinite loops
)

print("âœ… DocumentsReviewLoop created.")

#############################################################################
# Creating the Conformance Tester Agent
#############################################################################

conformance_tester_agent = SequentialAgent(
    name=AGENT_NAME,
    sub_agents=[parallel_learning_agent, learning_aggreg_agent, parallel_standard_analysis, consolidate_agent, formatting_agent, documents_review_loop_agent],
)

print(f"âœ… {AGENT_NAME} created.")

#############################################################################
# Executing the Conformance Tester Agent
#############################################################################

session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=conformance_tester_agent, app_name=APP_NAME, session_service=session_service)
response = await runner.run_debug(USER_PROMPT)



%%script echo skipping
# Comment the line above to execute this cell

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

url_prefix = get_adk_proxy_url()



%%script echo skipping
# Comment the line above to execute this cell

!adk web --log_level DEBUG --url_prefix {url_prefix}


%%script echo skipping
# Comment the line above to execute this cell

# Check the DEBUG logs from the broken agent
print("ğŸ”� Examining web server logs for debugging clues...\n")
!cat logger.log

